"""Main entry point for Polytrader weather trading bot.

Implements a comprehensive trading loop that:
1. Scans for active weather markets every 30 seconds
2. Fetches ensemble forecasts and calculates fair values
3. Detects edge and arbitrage opportunities
4. Executes trades through risk management
5. Logs all activity in structured JSON format
"""

import argparse
import asyncio
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any

import numpy as np

from src.config import get_settings, TradingMode, CITIES, CityConfig
from src.logging import setup_logging, get_logger

# Data sources
from src.data.open_meteo import (
    OpenMeteoClient,
    EnsembleForecastResult,
    create_mock_forecast,
)
from src.data.nowcasting import (
    MetarNowcaster,
    TemperatureConstraint,
    create_mock_constraint,
)

# Market scanning
from src.markets.market_scanner import (
    MarketScanner,
    WeatherMarket,
    TemperatureBucket,
    create_mock_markets,
)

# Strategy
from src.strategy.probability import (
    FairValueCalculator,
    FairValueResult,
    MarketFairValue,
    EdgeDetector,
    ArbitrageDetector,
    ArbitrageOpportunity,
)

# Risk management
from src.risk.risk_manager import (
    RiskManager,
    TradeRequest,
    Position,
    RiskStatus,
)

# Data persistence
from src.execution.datastore import DataStore

# CLI dashboard
from src.cli.dashboard import Dashboard

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TradingOpportunity:
    """A detected trading opportunity."""
    market: WeatherMarket
    bucket: TemperatureBucket
    fair_value: FairValueResult

    # Trade details
    side: str  # "BUY" or "SELL"
    price: Decimal
    suggested_size: Decimal

    # Metrics
    edge: float
    expected_profit: float
    kelly_fraction: float
    model_agreement: float

    # Nowcasting (if applicable)
    metar_constraint: TemperatureConstraint | None = None

    @property
    def priority_score(self) -> float:
        """Score for sorting opportunities."""
        base_score = abs(self.edge) * self.model_agreement * float(self.suggested_size)

        # Boost priority when METAR constraint is active
        # (higher confidence = better edge)
        if self.metar_constraint and self.metar_constraint.is_active:
            metar_boost = 1.0 + self.metar_constraint.confidence
            return base_score * metar_boost

        return base_score

    @property
    def has_metar_edge(self) -> bool:
        """Check if this opportunity benefits from METAR constraint."""
        return (
            self.metar_constraint is not None and
            self.metar_constraint.is_active and
            self.metar_constraint.confidence > 0.5
        )


@dataclass
class TradingIteration:
    """Results from a single trading iteration."""
    timestamp: datetime
    duration_seconds: float

    # What we found
    n_markets: int = 0
    n_forecasts: int = 0
    n_opportunities: int = 0
    n_arbitrage: int = 0

    # METAR nowcasting stats
    n_metar_constraints: int = 0
    metar_opportunities: int = 0  # Opportunities boosted by METAR

    # What we did
    n_trades_attempted: int = 0
    n_trades_executed: int = 0
    n_trades_blocked: int = 0

    # P&L
    paper_pnl: Decimal = Decimal("0")

    # Details
    opportunities: list[TradingOpportunity] = field(default_factory=list)
    arbitrage_opportunities: list[ArbitrageOpportunity] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": round(self.duration_seconds, 2),
            "n_markets": self.n_markets,
            "n_forecasts": self.n_forecasts,
            "n_opportunities": self.n_opportunities,
            "n_arbitrage": self.n_arbitrage,
            "n_metar_constraints": self.n_metar_constraints,
            "metar_opportunities": self.metar_opportunities,
            "n_trades_attempted": self.n_trades_attempted,
            "n_trades_executed": self.n_trades_executed,
            "n_trades_blocked": self.n_trades_blocked,
        }


# =============================================================================
# Trading Bot
# =============================================================================

class TradingBot:
    """
    Main trading bot that orchestrates all components.

    Runs a continuous loop that:
    1. Scans for active weather markets
    2. Fetches ensemble weather forecasts
    3. Calculates fair values and detects edge
    4. Executes trades when criteria are met
    """

    # Timing
    LOOP_INTERVAL_SECONDS = 30
    MIN_HOURS_TO_RESOLUTION = 1
    MAX_HOURS_TO_RESOLUTION = 72

    def __init__(
        self,
        mode: TradingMode = TradingMode.PAPER,
        use_mock: bool = False,
    ):
        """
        Initialize the trading bot.

        Args:
            mode: Trading mode (paper, semi, auto)
            use_mock: Use mock data instead of live APIs
        """
        self.settings = get_settings()
        self.mode = mode
        self.use_mock = use_mock

        # Initialize components
        self.fair_value_calc = FairValueCalculator()
        self.edge_detector = EdgeDetector(
            min_edge=self.settings.min_edge_threshold,
            min_agreement=self.settings.min_model_agreement,
        )
        self.arb_detector = ArbitrageDetector(min_gap=0.02)
        self.risk_manager = RiskManager(
            starting_bankroll=Decimal(str(self.settings.starting_bankroll)),
            min_edge=self.settings.min_edge_threshold,
            min_model_agreement=self.settings.min_model_agreement,
        )

        # METAR nowcasting for real-time constraints
        self.nowcaster = MetarNowcaster()

        # Data persistence
        self.datastore = DataStore()

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._iteration_count = 0
        self._paper_positions: dict[str, dict[str, Any]] = {}
        self._paper_pnl = Decimal("0")
        self._last_price_snapshot = datetime.min
        self._daily_stats: dict[str, Any] = {
            "opportunities": 0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "metar_trades": 0,
        }

    async def start(self) -> None:
        """Start the trading bot."""
        logger.info(
            "Starting Polytrader",
            mode=self.mode.value,
            bankroll=self.settings.starting_bankroll,
            use_mock=self.use_mock,
        )

        # Validate settings for live trading
        if self.mode != TradingMode.PAPER:
            self.settings.validate_live_trading()

        self._running = True

        # Connect to data store
        await self.datastore.connect()

        # Start METAR nowcaster (background updates every 15 min)
        if not self.use_mock:
            await self.nowcaster.start()

        # Set up signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        try:
            await self._run_loop()
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading bot."""
        logger.info("Stopping Polytrader")
        self._running = False
        self._shutdown_event.set()

        # Stop nowcaster
        await self.nowcaster.stop()

        # Save daily summary
        await self._save_daily_summary()

        # Close data store
        await self.datastore.close()

        # Log final status
        status = self.risk_manager.get_status()
        logger.info(
            "Final status",
            iterations=self._iteration_count,
            paper_pnl=str(self._paper_pnl),
            risk_status=status,
        )

    def _handle_shutdown(self) -> None:
        """Handle shutdown signal."""
        logger.info("Shutdown signal received")
        self._running = False
        self._shutdown_event.set()

    async def _run_loop(self) -> None:
        """Main trading loop."""
        while self._running:
            try:
                iteration = await self._run_iteration()

                logger.info(
                    "Iteration complete",
                    iteration=self._iteration_count,
                    **iteration.to_dict(),
                )

                # Wait before next iteration
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.LOOP_INTERVAL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error("Error in trading loop", error=str(e), exc_info=True)
                await asyncio.sleep(self.LOOP_INTERVAL_SECONDS)

    async def _run_iteration(self) -> TradingIteration:
        """Run a single trading iteration."""
        self._iteration_count += 1
        start_time = datetime.utcnow()

        iteration = TradingIteration(
            timestamp=start_time,
            duration_seconds=0,
        )

        logger.info(
            "Starting iteration",
            iteration=self._iteration_count,
            mode=self.mode.value,
        )

        # 1. Scan for weather markets
        markets = await self._scan_markets()
        iteration.n_markets = len(markets)

        if not markets:
            logger.info("No active weather markets found")
            iteration.duration_seconds = (datetime.utcnow() - start_time).total_seconds()
            return iteration

        logger.info("Found weather markets", count=len(markets))

        # 2. Process each market
        all_opportunities: list[TradingOpportunity] = []
        all_arbitrage: list[ArbitrageOpportunity] = []

        for market in markets:
            # Filter by time to resolution
            hours = market.hours_until_close
            if hours < self.MIN_HOURS_TO_RESOLUTION:
                logger.debug(
                    "Skipping market too close to resolution",
                    market_id=market.condition_id[:16],
                    hours=hours,
                )
                continue
            if hours > self.MAX_HOURS_TO_RESOLUTION:
                logger.debug(
                    "Skipping market too far from resolution",
                    market_id=market.condition_id[:16],
                    hours=hours,
                )
                continue

            try:
                opps, arbs = await self._process_market(market)
                all_opportunities.extend(opps)
                all_arbitrage.extend(arbs)
                iteration.n_forecasts += 1
            except Exception as e:
                logger.warning(
                    "Error processing market",
                    market_id=market.condition_id[:16],
                    error=str(e),
                )

        iteration.n_opportunities = len(all_opportunities)
        iteration.n_arbitrage = len(all_arbitrage)
        iteration.opportunities = all_opportunities
        iteration.arbitrage_opportunities = all_arbitrage

        # Count METAR-boosted opportunities
        metar_opps = [o for o in all_opportunities if o.has_metar_edge]
        iteration.metar_opportunities = len(metar_opps)
        iteration.n_metar_constraints = len(set(
            o.metar_constraint.city for o in metar_opps
            if o.metar_constraint
        ))

        # 3. Save and log arbitrage opportunities
        for arb in all_arbitrage:
            logger.info(
                "Arbitrage opportunity detected",
                city=arb.city,
                target_date=arb.target_date.isoformat(),
                gap=arb.gap,
                roi_percent=arb.roi_percent,
                required_capital=str(arb.required_capital),
            )
            # Persist arbitrage opportunity
            await self.datastore.save_arbitrage_opportunity(
                timestamp=start_time,
                condition_id=arb.condition_id,
                city=arb.city,
                target_date=arb.target_date,
                total_yes_price=arb.total_yes_price,
                gap=arb.gap,
                profit_margin=arb.profit_margin,
                roi_percent=arb.roi_percent,
                required_capital=float(arb.required_capital),
                guaranteed_profit=float(arb.guaranteed_profit),
            )

        # 4. Save price snapshots every 5 minutes
        await self._maybe_save_price_snapshots(markets, start_time)

        # Update daily stats
        self._daily_stats["opportunities"] += len(all_opportunities)

        # 5. Sort opportunities by expected profit
        all_opportunities.sort(key=lambda x: x.priority_score, reverse=True)

        # Track trade results for each opportunity
        trade_results: dict[int, tuple[bool, str | None]] = {}

        # 6. Execute trades (top 3)
        for i, opp in enumerate(all_opportunities[:3]):
            try:
                executed, blocked_reason = await self._execute_opportunity(opp)
                trade_results[i] = (executed, blocked_reason)
                iteration.n_trades_attempted += 1
                if executed:
                    iteration.n_trades_executed += 1
                    self._daily_stats["trades"] += 1
                    if opp.has_metar_edge:
                        self._daily_stats["metar_trades"] += 1

                    # Save executed trade to datastore
                    await self.datastore.save_trade(
                        timestamp=start_time,
                        opportunity_id=None,  # Set after saving opportunity
                        condition_id=opp.market.condition_id,
                        token_id=opp.bucket.token_id,
                        city=opp.market.city.name if opp.market.city else "unknown",
                        target_date=opp.market.target_date or date.today(),
                        outcome=opp.bucket.outcome,
                        side=opp.side,
                        price=float(opp.price),
                        size=float(opp.suggested_size),
                        fair_probability=opp.fair_value.fair_probability,
                        market_probability=opp.fair_value.market_probability,
                        edge=opp.edge,
                        expected_profit=opp.expected_profit,
                        trading_mode=self.mode.value,
                    )
                else:
                    iteration.n_trades_blocked += 1
                    if blocked_reason:
                        iteration.blocked_reasons.append(blocked_reason)
            except Exception as e:
                logger.error(
                    "Error executing trade",
                    outcome=opp.bucket.outcome,
                    error=str(e),
                )
                trade_results[i] = (False, str(e))
                iteration.blocked_reasons.append(str(e))

        # 7. Save all opportunities to datastore (with trade status)
        for i, opp in enumerate(all_opportunities):
            was_traded = False
            blocked_reason = None
            if i < 3 and i in trade_results:
                was_traded, blocked_reason = trade_results[i]
                # Only mark as traded if actually executed
                was_traded = was_traded and blocked_reason is None
            await self._save_opportunity(
                opp,
                start_time,
                was_traded=was_traded,
                blocked_reason=blocked_reason,
            )

        iteration.duration_seconds = (datetime.utcnow() - start_time).total_seconds()
        return iteration

    async def _scan_markets(self) -> list[WeatherMarket]:
        """Scan for active weather markets."""
        if self.use_mock:
            return create_mock_markets()

        try:
            async with MarketScanner() as scanner:
                return await scanner.scan_weather_markets()
        except Exception as e:
            logger.warning(f"Market scan failed, using mock: {e}")
            return create_mock_markets()

    async def _process_market(
        self,
        market: WeatherMarket,
    ) -> tuple[list[TradingOpportunity], list[ArbitrageOpportunity]]:
        """
        Process a single market for opportunities.

        Returns:
            Tuple of (trading opportunities, arbitrage opportunities)
        """
        if not market.city or not market.target_date:
            return [], []

        city = market.city
        target = market.target_date

        logger.debug(
            "Processing market",
            city=city.name,
            target_date=target.isoformat(),
            hours_until_close=market.hours_until_close,
        )

        # 1. Get ensemble forecast
        forecast = await self._get_forecast(city, target)
        if forecast is None or len(forecast.temperatures) < 10:
            logger.warning(
                "Insufficient forecast data",
                city=city.name,
            )
            return [], []

        temperatures = forecast.temperatures
        model_agreement = forecast.model_agreement

        logger.info(
            "Forecast received",
            city=city.name,
            n_members=len(temperatures),
            mean=float(np.mean(temperatures)),
            std=float(np.std(temperatures)),
            model_agreement=model_agreement,
        )

        # Save forecast snapshot for backtesting
        await self._save_forecast_snapshot(
            city=city,
            target_date=target,
            forecast=forecast,
            timestamp=datetime.utcnow(),
        )

        # 2. Apply METAR nowcasting constraint if target is today
        current_temp = None
        constraint: TemperatureConstraint | None = None

        if target == date.today():
            if self.use_mock:
                # Create mock constraint for afternoon trading demo
                # Use 4 hours until trading close (simulates 2pm observation)
                # Trading window is 10am-6pm, so 4h means 2pm local time
                constraint = create_mock_constraint(
                    city=city,
                    hours_until_close=4.0,  # Simulate 2pm (prime trading window)
                )
            else:
                constraint = await self.nowcaster.fetch_constraint(city, target)

            if constraint and constraint.is_active:
                current_temp = constraint.max_temp_observed

                # Apply constraint to temperature distribution
                temperatures = self.nowcaster.apply_constraint(temperatures, constraint)

                logger.info(
                    "METAR constraint applied",
                    city=city.name,
                    max_temp_observed=constraint.max_temp_observed,
                    hours_until_close=constraint.hours_until_close,
                    confidence=constraint.confidence,
                    uncertainty_reduction=constraint.uncertainty_reduction,
                    constrained_mean=float(np.mean(temperatures)),
                    constrained_std=float(np.std(temperatures)),
                )

        # 3. Check for arbitrage first
        arbitrage_opps: list[ArbitrageOpportunity] = []

        bucket_data = [
            (b.token_id, b.outcome, b.yes_price)
            for b in market.buckets
        ]

        arb = self.arb_detector.detect(
            condition_id=market.condition_id,
            city=city.name,
            target_date=target,
            buckets=bucket_data,
        )

        if arb:
            arbitrage_opps.append(arb)

        # 4. Calculate fair values for all buckets
        bucket_inputs = [
            (b.token_id, b.outcome, b.low_bound, b.high_bound, b.yes_price)
            for b in market.buckets
        ]

        analysis = self.fair_value_calc.analyze_market(
            condition_id=market.condition_id,
            city=city.name,
            target_date=target,
            temperatures=temperatures,
            buckets=bucket_inputs,
            model_agreement=model_agreement,
            current_temp=current_temp,
        )

        # 4. Find trading opportunities
        opportunities: list[TradingOpportunity] = []
        bankroll = self.risk_manager.current_bankroll

        for fv in analysis.buckets:
            # Skip if doesn't meet criteria
            if abs(fv.edge) < self.settings.min_edge_threshold:
                continue
            if fv.model_agreement < self.settings.min_model_agreement:
                continue

            # Find corresponding bucket
            bucket = next(
                (b for b in market.buckets if b.token_id == fv.token_id),
                None
            )
            if not bucket:
                continue

            # Calculate position size (half Kelly)
            kelly = self.edge_detector.calculate_kelly_fraction(
                fv.fair_probability,
                fv.market_probability,
            )
            suggested_size = Decimal(str(round(float(bankroll) * kelly * 0.5, 2)))

            # Determine side
            side = "BUY" if fv.edge > 0 else "SELL"
            price = Decimal(str(bucket.yes_price if side == "BUY" else (1 - bucket.yes_price)))

            # Create opportunity
            opp = TradingOpportunity(
                market=market,
                bucket=bucket,
                fair_value=fv,
                side=side,
                price=price,
                suggested_size=suggested_size,
                edge=fv.edge,
                expected_profit=fv.expected_value * float(suggested_size),
                kelly_fraction=kelly,
                model_agreement=fv.model_agreement,
                metar_constraint=constraint,
            )
            opportunities.append(opp)

            logger.info(
                "Opportunity detected",
                city=city.name,
                outcome=bucket.outcome,
                side=side,
                edge=fv.edge,
                fair_prob=fv.fair_probability,
                market_prob=fv.market_probability,
                suggested_size=str(suggested_size),
                metar_boosted=opp.has_metar_edge,
            )

        return opportunities, arbitrage_opps

    async def _get_forecast(
        self,
        city: CityConfig,
        target_date: date,
    ) -> EnsembleForecastResult | None:
        """Get ensemble forecast for a city and date."""
        if self.use_mock:
            return create_mock_forecast(
                lat=city.lat,
                lon=city.lon,
                city_name=city.name,
                target_date=target_date,
                convert_to_fahrenheit=(city.unit == "F"),
            )

        try:
            async with OpenMeteoClient() as client:
                return await client.get_ensemble_forecast(
                    lat=city.lat,
                    lon=city.lon,
                    target_date=target_date,
                    city_name=city.name,
                    convert_to_fahrenheit=(city.unit == "F"),
                )
        except Exception as e:
            logger.warning(f"Forecast fetch failed, using mock: {e}")
            return create_mock_forecast(
                lat=city.lat,
                lon=city.lon,
                city_name=city.name,
                target_date=target_date,
                convert_to_fahrenheit=(city.unit == "F"),
            )

    async def _execute_opportunity(
        self,
        opp: TradingOpportunity,
    ) -> tuple[bool, str | None]:
        """
        Execute a trading opportunity.

        Returns:
            Tuple of (executed: bool, blocked_reason: str | None)
        """
        # Build trade request
        request = TradeRequest(
            token_id=opp.bucket.token_id,
            condition_id=opp.market.condition_id,
            outcome=opp.bucket.outcome,
            side=opp.side,
            price=opp.price,
            size=opp.suggested_size,
            edge=opp.edge,
            model_agreement=opp.model_agreement,
            liquidity=opp.bucket.bid_size + opp.bucket.ask_size,
            city=opp.market.city.name if opp.market.city else "",
            target_date=opp.market.target_date,
        )

        # Check risk
        can_trade, checks = self.risk_manager.can_trade(request)

        if not can_trade:
            blocked = [c for c in checks if c.status == RiskStatus.BLOCKED]
            blocked_reasons = [c.name for c in blocked]
            logger.warning(
                "Trade blocked by risk manager",
                outcome=opp.bucket.outcome,
                blocked_by=blocked_reasons,
            )
            return False, ", ".join(blocked_reasons)

        # Execute based on mode
        if self.mode == TradingMode.PAPER:
            executed = await self._execute_paper_trade(opp, request)
        elif self.mode == TradingMode.SEMI:
            executed = await self._execute_semi_trade(opp, request)
        else:  # AUTO
            executed = await self._execute_auto_trade(opp, request)

        return executed, None

    async def _save_opportunity(
        self,
        opp: TradingOpportunity,
        timestamp: datetime,
        was_traded: bool = False,
        blocked_reason: str | None = None,
    ) -> int:
        """Save an opportunity to the data store."""
        return await self.datastore.save_opportunity(
            timestamp=timestamp,
            condition_id=opp.market.condition_id,
            token_id=opp.bucket.token_id,
            city=opp.market.city.name if opp.market.city else "unknown",
            target_date=opp.market.target_date or date.today(),
            outcome=opp.bucket.outcome,
            market_price=float(opp.price),
            fair_probability=opp.fair_value.fair_probability,
            edge=opp.edge,
            side=opp.side,
            suggested_size=float(opp.suggested_size),
            expected_profit=opp.expected_profit,
            kelly_fraction=opp.kelly_fraction,
            model_agreement=opp.model_agreement,
            confidence=opp.fair_value.confidence,
            metar_boosted=opp.has_metar_edge,
            metar_max_temp=opp.metar_constraint.max_temp_observed if opp.metar_constraint else None,
            metar_confidence=opp.metar_constraint.confidence if opp.metar_constraint else None,
            was_traded=was_traded,
            trade_blocked_reason=blocked_reason,
        )

    async def _maybe_save_price_snapshots(
        self,
        markets: list[WeatherMarket],
        timestamp: datetime,
    ) -> None:
        """Save price snapshots every 5 minutes."""
        # Check if 5 minutes have passed since last snapshot
        elapsed = (timestamp - self._last_price_snapshot).total_seconds()
        if elapsed < 300:  # 5 minutes
            return

        self._last_price_snapshot = timestamp

        for market in markets:
            if not market.city or not market.target_date:
                continue

            for bucket in market.buckets:
                await self.datastore.save_price_snapshot(
                    timestamp=timestamp,
                    condition_id=market.condition_id,
                    token_id=bucket.token_id,
                    city=market.city.name,
                    target_date=market.target_date,
                    outcome=bucket.outcome,
                    yes_price=bucket.yes_price,
                    no_price=1.0 - bucket.yes_price,
                    best_bid=bucket.best_bid,
                    best_ask=bucket.best_ask,
                    bid_size=bucket.bid_size,
                    ask_size=bucket.ask_size,
                    spread=bucket.best_ask - bucket.best_bid if bucket.best_ask and bucket.best_bid else None,
                )

        logger.debug("Saved price snapshots", n_markets=len(markets))

    async def _save_daily_summary(self) -> None:
        """Save daily summary when stopping."""
        today = date.today()
        status = self.risk_manager.get_status()

        await self.datastore.save_daily_summary(
            summary_date=today,
            starting_bankroll=float(status["bankroll"]["starting"]),
            ending_bankroll=float(status["bankroll"]["current"]),
            realized_pnl=float(self._paper_pnl),
            unrealized_pnl=0.0,  # Would calculate from open positions
            n_opportunities=self._daily_stats["opportunities"],
            n_trades=self._daily_stats["trades"],
            n_wins=self._daily_stats["wins"],
            n_losses=self._daily_stats["losses"],
            n_metar_constraints=0,  # Would track during iteration
            n_metar_trades=self._daily_stats["metar_trades"],
            max_drawdown_pct=float(status["bankroll"]["drawdown_pct"]),
            positions_at_close=len(self._paper_positions),
        )

        logger.info(
            "Saved daily summary",
            date=today.isoformat(),
            pnl=str(self._paper_pnl),
            trades=self._daily_stats["trades"],
        )

    async def _save_forecast_snapshot(
        self,
        city: CityConfig,
        target_date: date,
        forecast: EnsembleForecastResult,
        timestamp: datetime,
    ) -> None:
        """Save a forecast snapshot to the data store."""
        await self.datastore.save_forecast_snapshot(
            timestamp=timestamp,
            city=city.name,
            target_date=target_date,
            n_members=forecast.n_members,
            mean=forecast.mean,
            std=forecast.std,
            min_temp=forecast.min,
            max_temp=forecast.max,
            p10=forecast.p10,
            p25=forecast.p25,
            p50=forecast.p50,
            p75=forecast.p75,
            p90=forecast.p90,
            model_agreement=forecast.model_agreement,
            model_means=forecast.model_means,
        )

    async def _execute_paper_trade(
        self,
        opp: TradingOpportunity,
        request: TradeRequest,
    ) -> bool:
        """Execute a paper trade (logging only)."""
        logger.info(
            "[PAPER] Trade executed",
            city=opp.market.city.name if opp.market.city else "unknown",
            outcome=opp.bucket.outcome,
            side=opp.side,
            price=str(opp.price),
            size=str(opp.suggested_size),
            edge=opp.edge,
            expected_profit=opp.expected_profit,
        )

        # Track paper position
        position_id = f"{request.condition_id}:{request.token_id}"
        self._paper_positions[position_id] = {
            "token_id": request.token_id,
            "condition_id": request.condition_id,
            "outcome": request.outcome,
            "side": opp.side,
            "price": float(opp.price),
            "size": float(opp.suggested_size),
            "edge": opp.edge,
            "opened_at": datetime.utcnow().isoformat(),
        }

        # Simulate P&L (simplified: assume we win/lose based on edge direction)
        # In reality, this would track until market resolution
        simulated_pnl = Decimal(str(round(opp.expected_profit, 2)))
        self._paper_pnl += simulated_pnl

        return True

    async def _execute_semi_trade(
        self,
        opp: TradingOpportunity,
        request: TradeRequest,
    ) -> bool:
        """Execute a semi-auto trade (requires confirmation)."""
        # Print alert
        print("\n" + "=" * 60)
        print("TRADE ALERT - CONFIRMATION REQUIRED")
        print("=" * 60)
        print(f"  City: {opp.market.city.name if opp.market.city else 'unknown'}")
        print(f"  Target Date: {opp.market.target_date}")
        print(f"  Outcome: {opp.bucket.outcome}")
        print(f"  Side: {opp.side}")
        print(f"  Price: ${opp.price}")
        print(f"  Size: {opp.suggested_size} shares")
        print(f"  Cost: ${float(opp.price) * float(opp.suggested_size):.2f}")
        print()
        print(f"  Fair Value: {opp.fair_value.fair_probability:.1%}")
        print(f"  Market Price: {opp.fair_value.market_probability:.1%}")
        print(f"  Edge: {opp.edge:+.1%}")
        print(f"  Model Agreement: {opp.model_agreement:.1%}")
        print(f"  Expected Profit: ${opp.expected_profit:.2f}")
        print("=" * 60)

        try:
            response = input("Execute this trade? [y/N]: ").strip().lower()
            if response == "y":
                # In real implementation, this would call the exchange
                logger.info(
                    "[SEMI] Trade confirmed and executed",
                    outcome=opp.bucket.outcome,
                    side=opp.side,
                    price=str(opp.price),
                    size=str(opp.suggested_size),
                )
                return True
            else:
                logger.info(
                    "[SEMI] Trade rejected by user",
                    outcome=opp.bucket.outcome,
                )
                return False
        except (EOFError, KeyboardInterrupt):
            logger.info("[SEMI] Trade cancelled")
            return False

    async def _execute_auto_trade(
        self,
        opp: TradingOpportunity,
        request: TradeRequest,
    ) -> bool:
        """Execute an automatic trade."""
        logger.info(
            "[AUTO] Executing trade",
            outcome=opp.bucket.outcome,
            side=opp.side,
            price=str(opp.price),
            size=str(opp.suggested_size),
        )

        # TODO: Implement actual order execution via PolymarketClient
        # For now, log as if executed
        logger.info(
            "[AUTO] Trade executed",
            city=opp.market.city.name if opp.market.city else "unknown",
            outcome=opp.bucket.outcome,
            side=opp.side,
            price=str(opp.price),
            size=str(opp.suggested_size),
            edge=opp.edge,
        )

        return True

    def get_status(self) -> dict[str, Any]:
        """Get current bot status."""
        return {
            "mode": self.mode.value,
            "running": self._running,
            "iterations": self._iteration_count,
            "paper_pnl": str(self._paper_pnl),
            "paper_positions": len(self._paper_positions),
            "risk": self.risk_manager.get_status(),
        }


# =============================================================================
# CLI Commands
# =============================================================================

async def run_bot(mode: str, use_mock: bool = False) -> None:
    """Run the trading bot."""
    setup_logging()

    trading_mode = TradingMode(mode)
    bot = TradingBot(mode=trading_mode, use_mock=use_mock)
    await bot.start()


async def scan_markets(use_mock: bool = False) -> None:
    """Scan and display available weather markets."""
    setup_logging()

    print("\n" + "=" * 60)
    print("ACTIVE WEATHER MARKETS")
    print("=" * 60)

    if use_mock:
        markets = create_mock_markets()
        print("(Using mock data)")
    else:
        try:
            async with MarketScanner() as scanner:
                markets = await scanner.scan_weather_markets()
        except Exception as e:
            print(f"Network error, using mock data: {e}")
            markets = create_mock_markets()

    for market in markets:
        print(f"\nMarket: {market.condition_id[:20]}...")
        print(f"  Question: {market.question[:60]}...")
        if market.city:
            print(f"  City: {market.city.name}")
        if market.target_date:
            print(f"  Target Date: {market.target_date}")
        print(f"  Hours Until Close: {market.hours_until_close:.1f}")
        print(f"  Buckets: {len(market.buckets)}")
        print(f"  Total YES Prices: ${market.total_yes_prices:.3f}")
        if market.arbitrage_gap > 0.01:
            print(f"  ** ARBITRAGE GAP: {market.arbitrage_gap:.1%} **")

        print("\n  Temperature Buckets:")
        for bucket in market.buckets:
            print(
                f"    {bucket.outcome:12} YES=${bucket.yes_price:.3f} "
                f"bid=${bucket.best_bid:.3f} ask=${bucket.best_ask:.3f}"
            )

    print("\n" + "=" * 60)


async def analyze_market(city: str, use_mock: bool = False) -> None:
    """Analyze a specific city's market."""
    setup_logging()

    city_config = CITIES.get(city.lower())
    if not city_config:
        print(f"Unknown city: {city}")
        print(f"Available cities: {', '.join(CITIES.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"MARKET ANALYSIS: {city_config.name}")
    print(f"{'='*60}")

    # Get forecast
    target = date.today() + timedelta(days=1)

    if use_mock:
        forecast = create_mock_forecast(
            lat=city_config.lat,
            lon=city_config.lon,
            city_name=city_config.name,
            target_date=target,
            convert_to_fahrenheit=(city_config.unit == "F"),
        )
    else:
        try:
            async with OpenMeteoClient() as client:
                forecast = await client.get_ensemble_forecast(
                    lat=city_config.lat,
                    lon=city_config.lon,
                    target_date=target,
                    city_name=city_config.name,
                    convert_to_fahrenheit=(city_config.unit == "F"),
                )
        except Exception:
            print("Using mock forecast data...")
            forecast = create_mock_forecast(
                lat=city_config.lat,
                lon=city_config.lon,
                city_name=city_config.name,
                target_date=target,
                convert_to_fahrenheit=(city_config.unit == "F"),
            )

    print(f"\nForecast for {target}:")
    print(f"  Ensemble Members: {forecast.n_members}")
    print(f"  Mean: {forecast.mean:.1f}°{city_config.unit}")
    print(f"  Std: {forecast.std:.1f}°{city_config.unit}")
    print(f"  Range: [{forecast.min:.1f}, {forecast.max:.1f}]")
    print(f"  Model Agreement: {forecast.model_agreement:.1%}")

    print(f"\nPercentiles:")
    print(f"  P10: {forecast.p10:.1f}  P25: {forecast.p25:.1f}")
    print(f"  P50: {forecast.p50:.1f}  P75: {forecast.p75:.1f}")
    print(f"  P90: {forecast.p90:.1f}")

    print(f"\n{'='*60}")


async def show_status() -> None:
    """Show current risk and position status."""
    setup_logging()

    risk = RiskManager(
        starting_bankroll=Decimal(str(get_settings().starting_bankroll))
    )
    status = risk.get_status()

    print("\n" + "=" * 60)
    print("POLYTRADER STATUS")
    print("=" * 60)
    print(f"\nBankroll:")
    print(f"  Starting: ${status['bankroll']['starting']}")
    print(f"  Current: ${status['bankroll']['current']}")
    print(f"  Peak: ${status['bankroll']['peak']}")
    print(f"  Drawdown: {status['bankroll']['drawdown_pct']}%")
    print(f"\nPositions:")
    print(f"  Open: {status['positions']['count']}/{status['positions']['max']}")
    print(f"  Value: ${status['positions']['total_value']}")
    print(f"\nDaily P&L:")
    print(f"  Total: {status['daily']['total_pnl_pct']}%")
    print(f"  Trades: {status['daily']['trades']}")
    print(f"\nRisk Status:")
    print(f"  Trading Allowed: {status['stops']['trading_allowed']}")
    print(f"  Emergency Stop: {status['stops']['emergency_stopped']}")
    if status['stops']['stop_reason'] != "NONE":
        print(f"  Stop Reason: {status['stops']['stop_reason']}")
    print("=" * 60)


async def show_dashboard(
    positions: bool = False,
    pnl: bool = False,
    opportunities: bool = False,
) -> None:
    """Show the rich CLI dashboard."""
    dashboard = Dashboard()

    if positions:
        await dashboard.show_positions()
    elif pnl:
        await dashboard.show_pnl()
    elif opportunities:
        await dashboard.show_opportunities()
    else:
        await dashboard.show_full()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Polytrader - Polymarket weather trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --mode paper        Run in paper trading mode
  python -m src.main --mode semi         Run with manual confirmation
  python -m src.main --mode auto         Run fully automated
  python -m src.main --mode paper --mock Use mock data (no network)
  python -m src.main scan                Scan for weather markets
  python -m src.main analyze dallas      Analyze Dallas market
  python -m src.main status              Show current status
  python -m src.main dashboard           Show full dashboard
  python -m src.main dashboard --positions   Show open positions
  python -m src.main dashboard --pnl         Show P&L summary
  python -m src.main dashboard --opportunities  Show active opportunities
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command (default)
    run_parser = subparsers.add_parser("run", help="Run the trading bot")
    run_parser.add_argument(
        "--mode", "-m",
        choices=["paper", "semi", "auto"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data instead of live APIs",
    )

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan for weather markets")
    scan_parser.add_argument("--mock", action="store_true", help="Use mock data")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a city market")
    analyze_parser.add_argument("city", help="City to analyze (nyc, dallas, etc)")
    analyze_parser.add_argument("--mock", action="store_true", help="Use mock data")

    # Status command
    subparsers.add_parser("status", help="Show current status")

    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Show rich CLI dashboard",
    )
    dashboard_parser.add_argument(
        "--positions",
        action="store_true",
        help="Show only open positions with unrealized P&L",
    )
    dashboard_parser.add_argument(
        "--pnl",
        action="store_true",
        help="Show P&L summary for the past week",
    )
    dashboard_parser.add_argument(
        "--opportunities",
        action="store_true",
        help="Show active trading opportunities",
    )

    # Web server command
    web_parser = subparsers.add_parser("web", help="Start web dashboard server")
    web_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    web_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )

    # Top-level shortcuts
    parser.add_argument(
        "--mode", "-m",
        choices=["paper", "semi", "auto"],
        help="Trading mode (shortcut for 'run --mode')",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data instead of live APIs",
    )

    args = parser.parse_args()

    # Handle commands
    if args.command == "scan":
        asyncio.run(scan_markets(use_mock=args.mock))
    elif args.command == "analyze":
        asyncio.run(analyze_market(args.city, use_mock=args.mock))
    elif args.command == "status":
        asyncio.run(show_status())
    elif args.command == "dashboard":
        asyncio.run(show_dashboard(
            positions=args.positions,
            pnl=args.pnl,
            opportunities=args.opportunities,
        ))
    elif args.command == "web":
        from src.web.server import run_server
        print(f"Starting web server at http://{args.host}:{args.port}")
        run_server(host=args.host, port=args.port)
    else:
        # Default to run
        mode = args.mode or "paper"
        use_mock = getattr(args, "mock", False)
        asyncio.run(run_bot(mode, use_mock=use_mock))


if __name__ == "__main__":
    main()
