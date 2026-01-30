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
from src.data.tomorrow import TomorrowClient, TomorrowForecast

# Resolution tracking (for ML training)
from src.resolution.tracker import ResolutionTracker

# Market scanning
from src.markets.market_scanner import (
    MarketScanner,
    WeatherMarket,
    TemperatureBucket,
    create_mock_markets,
)
from src.markets.client import PolymarketClient, OrderSide

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

# Session management
from src.execution.sessions import get_session_manager, TradingSession

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

    # BUY NO support (when SELL is converted to BUY NO)
    is_buy_no: bool = False
    effective_token_id: str = ""  # Token to actually trade (YES or NO)

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

    # Timing (interval now configurable via SCAN_INTERVAL_SECONDS env var)
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

        # CRITICAL: Sync command-line mode to settings singleton
        # The PolymarketClient checks settings.is_paper_trading, not self.mode
        # Without this, --mode auto still executes paper trades!
        self.settings.trading_mode = mode

        # Initialize components
        self.fair_value_calc = FairValueCalculator()
        self.edge_detector = EdgeDetector(
            min_edge=self.settings.min_edge_threshold,
            min_agreement=self.settings.min_model_agreement,
        )
        self.arb_detector = ArbitrageDetector(min_gap=0.02)
        self.risk_manager = RiskManager(
            starting_bankroll=Decimal(str(self.settings.starting_bankroll)),
            max_position_pct=self.settings.max_position_pct,
            max_concurrent_positions=self.settings.max_concurrent_positions,
            min_edge=self.settings.min_edge_threshold,
            min_model_agreement=self.settings.min_model_agreement,
        )

        # METAR nowcasting for real-time constraints
        self.nowcaster = MetarNowcaster()

        # Tomorrow.io client for ML-enhanced forecasts (with caching)
        self.tomorrow_client = TomorrowClient()

        # Polymarket client for order execution
        self.poly_client = PolymarketClient()

        # Data persistence
        self.datastore = DataStore()

        # Resolution tracker for ML training (updates predictions with actual outcomes)
        self.resolution_tracker: ResolutionTracker | None = None  # Initialized after datastore connects

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._iteration_count = 0
        self._paper_positions: dict[str, dict[str, Any]] = {}
        self._paper_pnl = Decimal("0")
        self._last_price_snapshot = datetime.min
        self._last_position_refresh = datetime.min  # Track last position sync
        self._daily_stats: dict[str, Any] = {
            "opportunities": 0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "metar_trades": 0,
        }

    async def _load_existing_positions(self) -> None:
        """Load existing positions from Polymarket to prevent duplicate trades."""
        if self.mode == TradingMode.PAPER:
            logger.info("Paper trading mode - skipping position load")
            return

        try:
            positions = await self.poly_client.get_positions()

            if not positions:
                logger.info("No existing positions found on Polymarket")
                return

            loaded_count = 0
            skipped_dust = 0
            skipped_resolved = 0
            for pos in positions:
                # Skip resolved/closed positions - they shouldn't count toward position limits
                # Polymarket data API may return positions that have already resolved
                # Check explicit resolution indicators only
                is_resolved = (
                    pos.get("resolved") is True or
                    pos.get("closed") is True or
                    str(pos.get("status", "")).upper() in ("RESOLVED", "CLOSED", "SETTLED") or
                    str(pos.get("outcome_status", "")).upper() in ("RESOLVED", "CLOSED", "SETTLED") or
                    pos.get("redeemable") is True
                )

                # Also check if currentValue is 0 but we have shares - indicates resolved/worthless
                # This catches "Lost" positions where shares are worthless
                cur_value = pos.get("currentValue", None) or pos.get("curPrice", None)
                size = pos.get("size", 0) or pos.get("balance", 0)
                try:
                    size_float = float(size) if size else 0
                    cur_value_float = float(cur_value) if cur_value is not None else None
                except (ValueError, TypeError):
                    size_float = 0
                    cur_value_float = None

                # If we have shares but currentValue is 0, the position resolved worthless
                if size_float > 0 and cur_value_float == 0:
                    is_resolved = True

                if is_resolved:
                    skipped_resolved += 1
                    logger.info(
                        "Skipped resolved/closed position",
                        outcome=pos.get("outcome", "") or pos.get("title", "") or pos.get("name", ""),
                        cur_value=cur_value,
                        size=size,
                    )
                    continue

                # Extract position data from Polymarket data API response
                # Data API format: asset (token_id), size, avgPrice, currentValue, etc.
                token_id = (
                    pos.get("asset", "") or
                    pos.get("token_id", "") or
                    pos.get("tokenId", "") or
                    pos.get("token", "")
                )
                size = pos.get("size", 0) or pos.get("balance", 0) or pos.get("amount", 0)

                # Skip zero or negative positions
                try:
                    size_float = float(size) if size else 0
                except (ValueError, TypeError):
                    size_float = 0

                # Get price to calculate position value
                avg_price = pos.get("avgPrice", 0) or pos.get("avg_price", 0) or pos.get("averagePrice", 0) or 0
                cur_price = pos.get("price", 0) or pos.get("currentPrice", 0) or pos.get("curPrice", 0) or avg_price
                try:
                    price_float = float(cur_price) if cur_price else 0
                except (ValueError, TypeError):
                    price_float = 0

                # Calculate position value - skip dust positions (< $1)
                position_value = size_float * price_float
                MIN_POSITION_VALUE = 1.0  # Ignore positions worth less than $1

                if token_id and size_float > 0 and position_value >= MIN_POSITION_VALUE:
                    # Create a Position object for the risk manager
                    position = Position(
                        token_id=str(token_id),
                        condition_id=pos.get("condition_id", "") or pos.get("conditionId", "") or pos.get("marketId", "") or "",
                        outcome=pos.get("outcome", "") or pos.get("title", "") or pos.get("name", "") or "unknown",
                        city="unknown",  # We don't have city info from API
                        target_date=None,
                        size=Decimal(str(size_float)),
                        entry_price=Decimal(str(avg_price)),
                        current_price=Decimal(str(cur_price)),
                        opened_at=datetime.utcnow(),
                    )
                    self.risk_manager._positions[token_id] = position
                    loaded_count += 1
                    logger.debug(
                        "Loaded position",
                        token_id=token_id[:20] + "...",
                        size=size_float,
                        outcome=position.outcome,
                    )
                elif token_id and size_float > 0 and position_value < MIN_POSITION_VALUE:
                    # Dust position - skip but count it
                    skipped_dust += 1
                    logger.debug(
                        "Skipped dust position",
                        token_id=token_id[:20] + "...",
                        value=round(position_value, 4),
                    )

            # Log all positions for visibility
            position_details = []
            for tid, pos in self.risk_manager._positions.items():
                position_details.append({
                    "outcome": pos.outcome[:30] if pos.outcome else "?",
                    "size": float(pos.size),
                    "value": round(float(pos.size * pos.current_price), 2),
                })

            logger.info(
                "Loaded existing positions from Polymarket",
                position_count=loaded_count,
                max_positions=self.settings.max_concurrent_positions,
                slots_available=self.settings.max_concurrent_positions - loaded_count,
                skipped_dust=skipped_dust,
                skipped_resolved=skipped_resolved,
            )

            # Log each position individually so user can see what's counted
            for detail in position_details:
                logger.info(
                    "Position loaded",
                    outcome=detail["outcome"],
                    size=detail["size"],
                    value=f"${detail['value']:.2f}",
                )

        except Exception as e:
            logger.warning(
                "Failed to load existing positions - will skip duplicate check for prior positions",
                error=str(e),
            )

    async def _refresh_positions_from_polymarket(self) -> None:
        """
        Sync internal position state with Polymarket.

        This detects when positions have resolved (been paid out) and removes
        them from the risk manager, freeing up slots for new trades.
        Called periodically (every 30 minutes) during operation.
        """
        if self.mode == TradingMode.PAPER:
            return

        try:
            # Fetch current positions from Polymarket
            live_positions = await self.poly_client.get_positions()

            # Build set of token_ids that still have active positions
            MIN_POSITION_VALUE = 1.0
            active_token_ids: set[str] = set()

            for pos in live_positions or []:
                # Skip resolved/closed positions
                is_resolved = (
                    pos.get("resolved") is True or
                    pos.get("closed") is True or
                    str(pos.get("status", "")).upper() in ("RESOLVED", "CLOSED", "SETTLED") or
                    str(pos.get("outcome_status", "")).upper() in ("RESOLVED", "CLOSED", "SETTLED") or
                    pos.get("redeemable") is True
                )

                # Also check if currentValue is 0 but we have shares - indicates resolved/worthless
                cur_value = pos.get("currentValue", None) or pos.get("curPrice", None)
                size_raw = pos.get("size", 0) or pos.get("balance", 0)
                try:
                    size_float = float(size_raw) if size_raw else 0
                    cur_value_float = float(cur_value) if cur_value is not None else None
                except (ValueError, TypeError):
                    size_float = 0
                    cur_value_float = None

                if size_float > 0 and cur_value_float == 0:
                    is_resolved = True

                if is_resolved:
                    continue

                token_id = (
                    pos.get("asset", "") or
                    pos.get("token_id", "") or
                    pos.get("tokenId", "") or
                    pos.get("token", "")
                )
                size = float(pos.get("size", 0) or pos.get("balance", 0) or 0)
                price = float(pos.get("price", 0) or pos.get("curPrice", 0) or pos.get("avgPrice", 0) or 0)

                if token_id and size > 0 and (size * price) >= MIN_POSITION_VALUE:
                    active_token_ids.add(token_id)

            # Find positions in risk manager that no longer exist on Polymarket
            internal_token_ids = set(self.risk_manager._positions.keys())
            resolved_token_ids = internal_token_ids - active_token_ids

            if resolved_token_ids:
                for token_id in resolved_token_ids:
                    pos = self.risk_manager._positions.pop(token_id, None)
                    if pos:
                        logger.info(
                            "Position resolved - slot freed",
                            outcome=pos.outcome,
                            token_id=token_id[:20] + "...",
                        )

                logger.info(
                    "Position refresh complete",
                    resolved_count=len(resolved_token_ids),
                    active_count=len(self.risk_manager._positions),
                    max_positions=self.settings.max_concurrent_positions,
                    slots_available=self.settings.max_concurrent_positions - len(self.risk_manager._positions),
                )

            self._last_position_refresh = datetime.utcnow()

        except Exception as e:
            logger.warning(
                "Failed to refresh positions from Polymarket",
                error=str(e),
            )

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

        # Initialize resolution tracker (needs datastore to be connected)
        self.resolution_tracker = ResolutionTracker(self.datastore)

        # Load existing positions from Polymarket to prevent duplicate trades
        await self._load_existing_positions()

        # Start METAR nowcaster (background updates every 15 min)
        if not self.use_mock:
            await self.nowcaster.start()

        # Set up signal handlers (Unix only - Windows doesn't support this)
        import sys
        if sys.platform != "win32":
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

                # Wait before next iteration (configurable via SCAN_INTERVAL_SECONDS)
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.settings.scan_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error("Error in trading loop", error=str(e), exc_info=True)
                await asyncio.sleep(self.settings.scan_interval_seconds)

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

        # 0. Refresh positions from Polymarket every 30 minutes
        # This detects resolved positions and frees up slots
        POSITION_REFRESH_INTERVAL = 1800  # 30 minutes in seconds
        time_since_refresh = (start_time - self._last_position_refresh).total_seconds()
        if time_since_refresh >= POSITION_REFRESH_INTERVAL:
            await self._refresh_positions_from_polymarket()

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

        # 5. Filter and convert opportunities to tradeable orders
        # - BUY opportunities: trade directly (BUY YES)
        # - SELL opportunities: convert to BUY NO (if no_token_id available)
        tradeable_opportunities: list[TradingOpportunity] = []
        n_sells_converted = 0
        n_sells_skipped = 0
        n_penny_picking_skipped = 0  # Trades skipped due to bad risk/reward

        # Maximum NO price to accept - avoid "penny in front of steamroller" trades
        # At 90¢, you risk $0.90 to make $0.10 = 9:1 risk/reward (terrible)
        # At 95¢, you risk $0.95 to make $0.05 = 19:1 risk/reward (awful)
        MAX_NO_PRICE = 0.90
        n_duplicate_skipped = 0

        for opp in all_opportunities:
            if opp.side == "BUY":
                # BUY YES - use the YES token
                opp.effective_token_id = opp.bucket.token_id
                opp.is_buy_no = False

                # Check if we already have a position in this token
                if self.risk_manager.has_position(opp.effective_token_id):
                    logger.debug(
                        "Skipping duplicate position",
                        outcome=opp.bucket.outcome,
                        token_id=opp.effective_token_id[:16],
                    )
                    n_duplicate_skipped += 1
                    continue

                tradeable_opportunities.append(opp)
            elif opp.side == "SELL":
                # SELL YES opportunity - convert to BUY NO
                no_token_id = getattr(opp.bucket, 'no_token_id', '')

                # Get actual NO price from orderbook (no_best_ask is the price to BUY NO)
                # CRITICAL: Do NOT use 1 - yes_price as fallback - on illiquid markets
                # YES + NO prices do NOT equal $1.00! This caused orders at wrong prices.
                no_best_ask = getattr(opp.bucket, 'no_best_ask', 0.0)

                if no_best_ask > 0:
                    # Use actual NO ask price from orderbook
                    no_price = no_best_ask
                else:
                    # No orderbook data for NO token - skip this trade
                    # We can't trust the calculated price
                    logger.debug(
                        "Skipping BUY NO - no orderbook data for NO token",
                        outcome=opp.bucket.outcome,
                        no_token_id=no_token_id[:16] if no_token_id else "none",
                    )
                    n_sells_skipped += 1
                    continue

                # Skip if NO price is too high (bad risk/reward)
                # Buying NO at 99¢ means risking $0.99 to potentially make $0.01
                if no_price > MAX_NO_PRICE:
                    logger.info(
                        "Skipping BUY NO - penny picking (bad risk/reward)",
                        outcome=opp.bucket.outcome,
                        no_price=f"${no_price:.3f}",
                        max_allowed=f"${MAX_NO_PRICE:.2f}",
                        risk_reward=f"{no_price/(1-no_price):.1f}:1",
                    )
                    n_penny_picking_skipped += 1
                    continue

                if no_token_id:
                    # Check if we already have a position in this NO token
                    if self.risk_manager.has_position(no_token_id):
                        logger.debug(
                            "Skipping duplicate NO position",
                            outcome=opp.bucket.outcome,
                            token_id=no_token_id[:16],
                        )
                        n_duplicate_skipped += 1
                        continue

                    # Convert: SELL YES at price P -> BUY NO at price (1-P)
                    # The edge is the same magnitude but we're buying underpriced NO
                    opp.side = "BUY"
                    opp.price = Decimal(str(no_price))
                    opp.effective_token_id = no_token_id
                    opp.is_buy_no = True
                    tradeable_opportunities.append(opp)
                    n_sells_converted += 1
                else:
                    # No NO token available, skip
                    n_sells_skipped += 1

        if n_sells_converted > 0 or n_sells_skipped > 0 or n_penny_picking_skipped > 0 or n_duplicate_skipped > 0:
            logger.info(
                "Processed opportunities",
                converted_to_buy_no=n_sells_converted,
                skipped_no_token=n_sells_skipped,
                skipped_penny_picking=n_penny_picking_skipped,
                skipped_duplicate=n_duplicate_skipped,
            )

        # Sort by expected profit
        tradeable_opportunities.sort(key=lambda x: x.priority_score, reverse=True)

        # Track trade results for each opportunity
        trade_results: dict[int, tuple[bool, str | None]] = {}

        # 6. Execute trades (top 3)
        for i, opp in enumerate(tradeable_opportunities[:3]):
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

        # 8. Check for market resolutions (for ML training)
        # This updates prediction records with actual outcomes from Wunderground.
        # The tracker has internal rate limiting (every 6 hours by default).
        if self.resolution_tracker:
            try:
                resolved = await self.resolution_tracker.check_resolutions()
                if resolved > 0:
                    logger.info(
                        "Resolved market outcomes",
                        resolved_count=resolved,
                    )
            except Exception as e:
                logger.warning(
                    "Resolution check failed",
                    error=str(e),
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

        # FILTER: Only trade same-day or next-day markets
        # Weather forecasts are most accurate 0-24 hours out
        # Beyond 2 days, ensemble spread is too wide for reliable edge
        MAX_DAYS_AHEAD = 1  # 0 = today, 1 = tomorrow
        days_ahead = (target - date.today()).days
        if days_ahead > MAX_DAYS_AHEAD:
            logger.debug(
                "Skipping market - too far in future",
                city=city.name,
                target_date=target.isoformat(),
                days_ahead=days_ahead,
                max_days=MAX_DAYS_AHEAD,
            )
            return [], []

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
        # Include bid/ask prices for proper edge calculation:
        # - BUY edge uses ask price (what we pay to buy)
        # - SELL edge uses bid price (what we receive when selling)
        bucket_inputs = [
            (b.token_id, b.outcome, b.low_bound, b.high_bound, b.yes_price, b.best_bid, b.best_ask)
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

        # 4a. Record ALL predictions for ML training (before filtering)
        prediction_ids = await self._record_predictions(
            market=market,
            forecast=forecast,
            analysis=analysis,
            constraint=constraint,
        )

        # 4b. Find trading opportunities
        opportunities: list[TradingOpportunity] = []
        bankroll = self.risk_manager.current_bankroll

        for fv in analysis.buckets:
            # Skip if doesn't meet criteria
            if abs(fv.edge) < self.settings.min_edge_threshold:
                continue
            if fv.model_agreement < self.settings.min_model_agreement:
                continue

            # SANITY CHECK: Reject unrealistic edges (>100% is almost certainly a data bug)
            # Real arbitrage opportunities are rarely > 50%, and anything > 100% indicates
            # stale orderbook data, unit mismatches, or API errors
            # Previous analysis showed trades with 200%+ edges had -4% P&L
            MAX_REALISTIC_EDGE = 1.0  # 100%
            if abs(fv.edge) > MAX_REALISTIC_EDGE:
                logger.warning(
                    "REJECTING SUSPICIOUS EDGE - likely data issue",
                    outcome=fv.outcome,
                    edge=f"{fv.edge:.1%}",
                    fair_prob=f"{fv.fair_probability:.1%}",
                    market_prob=f"{fv.market_probability:.1%}",
                    bucket_low=fv.low_bound,
                    bucket_high=fv.high_bound,
                    forecast_mean=f"{fv.kde_mean:.1f}",
                )
                print(f"\n⚠️  WARNING: Rejecting suspicious edge for {fv.outcome}")
                print(f"   Edge: {fv.edge:.1%} (>100% suggests data bug)")
                print(f"   Fair Value: {fv.fair_probability:.1%}")
                print(f"   Market Price: {fv.market_probability:.1%}")
                print(f"   Forecast Mean: {fv.kde_mean:.1f}°")
                print(f"   Bucket: {fv.low_bound} to {fv.high_bound}")
                continue

            # FILTER: Avoid long-shot bets (fair value < 10%)
            # Betting on <10% probability outcomes has high variance and poor expected growth
            MIN_FAIR_VALUE = 0.10  # 10%
            if fv.fair_probability < MIN_FAIR_VALUE:
                logger.debug(
                    "Skipping long-shot bet",
                    outcome=fv.outcome,
                    fair_prob=f"{fv.fair_probability:.1%}",
                )
                continue

            # FILTER: Never buy YES at high prices (penny-picking trap)
            # Jan 24 trades showed buying YES at $0.95+ with 0% prob = guaranteed loss
            # Max 70% ensures decent risk/reward (risk $0.70 to win $0.30)
            MAX_BUY_PRICE = 0.70  # 70%
            if fv.edge > 0 and fv.market_probability > MAX_BUY_PRICE:
                logger.debug(
                    "Skipping expensive YES (bad risk/reward)",
                    outcome=fv.outcome,
                    market_prob=f"{fv.market_probability:.1%}",
                )
                continue

            # FILTER: Minimum entry price for YES bets (avoid penny stock long-shots)
            # Trade analysis showed many losses from 5-10¢ YES bets on unlikely buckets
            # These have high "edge" but rarely hit - the tails are too uncertain
            MIN_YES_PRICE = 0.08  # 8% - bucket must be reasonably probable
            if fv.edge > 0 and fv.market_probability < MIN_YES_PRICE:
                logger.debug(
                    "Skipping cheap YES (penny stock long-shot)",
                    outcome=fv.outcome,
                    market_prob=f"{fv.market_probability:.1%}",
                    min_required=f"{MIN_YES_PRICE:.0%}",
                )
                continue

            # FILTER: YES bets must be within 3°F of forecast mean
            # CRITICAL INSIGHT from trade analysis: WINNING YES bets were on buckets
            # NEAR the forecast mean (e.g., Atlanta 44-45°F when mean was ~44°F)
            # LOSING YES bets were on buckets FAR from mean (cheap long-shots)
            MAX_DISTANCE_FOR_YES = 3.0  # Must be within 3°F of forecast mean
            if fv.edge > 0:  # BUY YES signal
                bucket_mid = (fv.low_bound + fv.high_bound) / 2
                distance_from_mean = abs(bucket_mid - fv.kde_mean)
                if distance_from_mean > MAX_DISTANCE_FOR_YES:
                    logger.debug(
                        "Skipping YES bet - bucket too far from forecast mean",
                        outcome=fv.outcome,
                        bucket_mid=f"{bucket_mid:.1f}",
                        forecast_mean=f"{fv.kde_mean:.1f}",
                        distance=f"{distance_from_mean:.1f}",
                        max_distance=MAX_DISTANCE_FOR_YES,
                    )
                    continue

            # FILTER: Disable NO bets entirely (trade analysis showed -76.5% P&L)
            # NO bets are complex: you're betting the bucket WON'T hit, but
            # the bot was buying NO on buckets that WERE near the forecast mean
            # Until YES bet logic is profitable, disable NO bets completely
            DISABLE_NO_BETS = True  # Set to False to re-enable NO bets
            if DISABLE_NO_BETS and fv.edge < 0:
                logger.debug(
                    "Skipping NO bet (NO bets disabled due to poor performance)",
                    outcome=fv.outcome,
                )
                continue

            # FILTER: Never buy NO at high prices (same penny-picking trap)
            # If NO costs 80¢, you risk 80¢ to win 20¢ - terrible risk/reward
            # Jan 28 analysis showed buying NO at 82-85¢ on likely outcomes = losses
            MAX_NO_PRICE = 0.70  # 70% - same limit as YES
            if fv.edge < 0:  # Negative edge = SELL YES = BUY NO
                no_price = 1 - fv.market_probability  # NO price ≈ 1 - YES price
                if no_price > MAX_NO_PRICE:
                    logger.debug(
                        "Skipping expensive NO (bad risk/reward)",
                        outcome=fv.outcome,
                        no_price=f"{no_price:.1%}",
                    )
                    continue

            # FILTER: Don't bet against buckets near the forecast mean
            # If forecast mean is 44°F and bucket is 44-45°F, don't buy NO!
            # Only buy NO on buckets that are far from the expected temperature
            if fv.edge < 0:  # BUY NO signal
                bucket_mid = (fv.low_bound + fv.high_bound) / 2
                distance_from_mean = abs(bucket_mid - fv.kde_mean)
                MIN_DISTANCE_FOR_NO = 3.0  # Must be 3+ degrees away from mean
                if distance_from_mean < MIN_DISTANCE_FOR_NO:
                    logger.debug(
                        "Skipping NO bet - bucket too close to forecast mean",
                        outcome=fv.outcome,
                        bucket_mid=bucket_mid,
                        forecast_mean=f"{fv.kde_mean:.1f}",
                        distance=f"{distance_from_mean:.1f}",
                    )
                    continue

                # FILTER: Only buy NO if the outcome is actually unlikely
                # If fair_value is 25%+, the outcome has decent chance - don't bet against it
                MAX_FAIR_VALUE_FOR_NO = 0.20  # Only bet NO if outcome <20% likely
                if fv.fair_probability > MAX_FAIR_VALUE_FOR_NO:
                    logger.debug(
                        "Skipping NO bet - outcome too likely",
                        outcome=fv.outcome,
                        fair_prob=f"{fv.fair_probability:.1%}",
                        max_for_no=f"{MAX_FAIR_VALUE_FOR_NO:.0%}",
                    )
                    continue

            # Find corresponding bucket
            bucket = next(
                (b for b in market.buckets if b.token_id == fv.token_id),
                None
            )
            if not bucket:
                continue

            # Calculate position size (half Kelly, capped at max_position_pct)
            kelly = self.edge_detector.calculate_kelly_fraction(
                fv.fair_probability,
                fv.market_probability,
            )
            half_kelly = kelly * 0.5
            max_position = float(bankroll) * self.settings.max_position_pct
            kelly_position = float(bankroll) * half_kelly
            suggested_size = Decimal(str(round(min(kelly_position, max_position), 2)))

            # Determine side and use correct orderbook price
            # fv.market_probability now contains the correct execution price:
            # - For BUY signals: this is the ask price
            # - For SELL signals: this is the bid price
            side = "BUY" if fv.edge > 0 else "SELL"
            if side == "BUY":
                # Use the execution price (ask price from edge calculation)
                price = Decimal(str(fv.market_probability))
            else:
                # For SELL YES (becomes BUY NO), use NO token ask if available
                # Otherwise calculate from YES bid: NO price ≈ 1 - YES bid
                if bucket.no_best_ask > 0:
                    price = Decimal(str(bucket.no_best_ask))
                else:
                    price = Decimal(str(1 - fv.market_probability))

            # Polymarket requires minimum 5 shares - ensure suggested_size is enough
            # min_dollars = 5 shares * price
            MIN_SHARES = Decimal("5")
            min_dollars_for_min_shares = MIN_SHARES * price
            if suggested_size < min_dollars_for_min_shares and suggested_size > 0:
                suggested_size = min_dollars_for_min_shares

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
        """Get ensemble forecast for a city and date, enhanced with Tomorrow.io."""
        if self.use_mock:
            return create_mock_forecast(
                lat=city.lat,
                lon=city.lon,
                city_name=city.name,
                target_date=target_date,
                convert_to_fahrenheit=(city.unit == "F"),
            )

        try:
            # Fetch Open-Meteo ensemble forecast (primary source)
            async with OpenMeteoClient() as client:
                forecast = await client.get_ensemble_forecast(
                    lat=city.lat,
                    lon=city.lon,
                    target_date=target_date,
                    city_name=city.name,
                    convert_to_fahrenheit=(city.unit == "F"),
                )

            if forecast is None:
                return None

            # Try to enhance with Tomorrow.io (optional, cached)
            await self._enhance_with_tomorrow(forecast, city, target_date)

            return forecast

        except Exception as e:
            # CRITICAL: Do NOT fall back to mock data for real trading
            # Mock data generates random temps (~55°F for all cities) which
            # creates false edges. Skip this market instead.
            logger.error(
                f"Forecast fetch failed - SKIPPING MARKET (not using mock data): {e}"
            )
            return None

    async def _enhance_with_tomorrow(
        self,
        forecast: EnsembleForecastResult,
        city: CityConfig,
        target_date: date,
    ) -> None:
        """Enhance ensemble forecast with Tomorrow.io ML forecast."""
        if not self.tomorrow_client.is_available:
            return

        # Maximum disagreement threshold (in forecast units)
        # If Tomorrow.io differs by more than this, skip blending
        MAX_DISAGREEMENT_F = 5.0  # 5°F for Fahrenheit cities
        MAX_DISAGREEMENT_C = 3.0  # ~3°C for Celsius cities

        try:
            tomorrow = await self.tomorrow_client.get_forecast(city, target_date)

            if tomorrow is None:
                return

            # Check if Tomorrow.io strongly disagrees with ensemble
            disagreement = abs(tomorrow.high_temp - forecast.mean)
            max_disagreement = MAX_DISAGREEMENT_F if city.unit == "F" else MAX_DISAGREEMENT_C

            if disagreement > max_disagreement:
                # Don't blend - the disagreement is too large
                # One source is likely wrong, don't let single ML forecast override 71 members
                logger.warning(
                    "Tomorrow.io strongly disagrees with ensemble - skipping blend",
                    city=city.name,
                    target_date=str(target_date),
                    tomorrow_high=tomorrow.high_temp,
                    ensemble_mean=forecast.mean,
                    disagreement=disagreement,
                    max_allowed=max_disagreement,
                )
                return

            # Add Tomorrow.io as synthetic ensemble members with tight spread
            # Tomorrow.io is ML-enhanced, so use smaller std dev (~2°F)
            tomorrow_samples = np.random.normal(
                tomorrow.high_temp,
                2.0,  # Tighter spread than raw NWP models
                20,   # Add 20 synthetic members
            )

            # Blend into existing ensemble
            original_temps = forecast.temperatures
            forecast.temperatures = np.concatenate([original_temps, tomorrow_samples])

            # Recalculate statistics with blended data
            forecast.mean = float(np.mean(forecast.temperatures))
            forecast.std = float(np.std(forecast.temperatures))
            forecast.min = float(np.min(forecast.temperatures))
            forecast.max = float(np.max(forecast.temperatures))
            forecast.p10 = float(np.percentile(forecast.temperatures, 10))
            forecast.p25 = float(np.percentile(forecast.temperatures, 25))
            forecast.p50 = float(np.percentile(forecast.temperatures, 50))
            forecast.p75 = float(np.percentile(forecast.temperatures, 75))
            forecast.p90 = float(np.percentile(forecast.temperatures, 90))

            logger.info(
                "Enhanced forecast with Tomorrow.io",
                city=city.name,
                target_date=str(target_date),
                tomorrow_high=tomorrow.high_temp,
                ensemble_mean=round(forecast.mean, 1),
                disagreement=round(disagreement, 1),
                original_members=len(original_temps),
                blended_members=len(forecast.temperatures),
                blended_mean=forecast.mean,
                blended_std=forecast.std,
            )

        except Exception as e:
            # Tomorrow.io enhancement is optional - don't fail the whole forecast
            logger.warning(
                "Failed to enhance forecast with Tomorrow.io",
                city=city.name,
                error=str(e),
            )

    async def _record_predictions(
        self,
        market: WeatherMarket,
        forecast: EnsembleForecastResult,
        analysis: MarketFairValue,
        constraint: TemperatureConstraint | None,
    ) -> dict[str, int]:
        """
        Record ALL bucket predictions for ML training.

        Records every bucket's model probability vs market probability,
        not just the ones we trade. Essential for calibration analysis.

        Returns:
            Dict mapping token_id -> prediction_record_id
        """
        prediction_ids: dict[str, int] = {}

        if not market.city or not market.target_date:
            return prediction_ids

        city = market.city
        target_date = market.target_date
        now = datetime.utcnow()

        # Check if Tomorrow.io was blended (check if we have more than 71 members)
        tomorrow_blended = len(forecast.temperatures) > 71

        for fv in analysis.buckets:
            # Find corresponding bucket for bounds
            bucket = next(
                (b for b in market.buckets if b.token_id == fv.token_id),
                None
            )

            try:
                pred_id = await self.datastore.save_prediction_record(
                    timestamp=now,
                    condition_id=market.condition_id,
                    token_id=fv.token_id,
                    city=city.name,
                    target_date=target_date,
                    outcome=fv.outcome,
                    bucket_low=fv.low_bound,
                    bucket_high=fv.high_bound,
                    unit=city.unit,
                    model_probability=fv.fair_probability,
                    market_probability=fv.market_probability,
                    calculated_edge=fv.edge,
                    ensemble_mean=forecast.mean,
                    ensemble_std=forecast.std,
                    ensemble_n_members=len(forecast.temperatures),
                    model_agreement=fv.model_agreement,
                    hours_to_resolution=market.hours_until_close,
                    ensemble_p10=forecast.p10,
                    ensemble_p50=forecast.p50,
                    ensemble_p90=forecast.p90,
                    model_means=getattr(forecast, 'model_means', None),
                    tomorrow_io_high=getattr(forecast, 'tomorrow_high', None),
                    tomorrow_io_blended=tomorrow_blended,
                    metar_max_temp=constraint.max_temp_observed if constraint else None,
                    metar_hours_remaining=constraint.hours_until_close if constraint else None,
                )
                prediction_ids[fv.token_id] = pred_id

            except Exception as e:
                logger.warning(
                    "Failed to save prediction record",
                    outcome=fv.outcome,
                    error=str(e),
                )

        if prediction_ids:
            logger.debug(
                "Recorded predictions for ML training",
                city=city.name,
                target_date=str(target_date),
                n_predictions=len(prediction_ids),
            )

        return prediction_ids

    async def _execute_opportunity(
        self,
        opp: TradingOpportunity,
    ) -> tuple[bool, str | None]:
        """
        Execute a trading opportunity.

        Returns:
            Tuple of (executed: bool, blocked_reason: str | None)
        """
        # Cap position size to max_position_pct of bankroll
        # This ensures SELL->BUY NO conversions don't get blocked due to uncapped sizes
        max_position_size = Decimal(str(self.settings.starting_bankroll)) * Decimal(str(self.settings.max_position_pct))
        capped_size = min(opp.suggested_size, max_position_size)

        # Also ensure minimum viable size ($1)
        if capped_size < Decimal("1"):
            logger.debug(
                "Position size too small after capping",
                outcome=opp.bucket.outcome,
                original_size=str(opp.suggested_size),
                capped_size=str(capped_size),
            )
            return False, "position_size_too_small"

        # Price freshness check - fetch current price and compare
        # This prevents placing orders at stale prices when market has moved
        if self.mode != TradingMode.PAPER:
            current_price = await self.poly_client.get_current_price(opp.bucket.token_id)
            if current_price is not None:
                # Calculate how much the price has moved
                order_price = float(opp.price)
                price_diff = abs(current_price - order_price)
                price_move_pct = price_diff / max(order_price, 0.01)

                # Skip if price moved more than 20% - edge likely gone
                MAX_PRICE_DRIFT = 0.20
                if price_move_pct > MAX_PRICE_DRIFT:
                    logger.warning(
                        "Skipping stale order - price moved significantly",
                        outcome=opp.bucket.outcome,
                        order_price=f"{order_price:.3f}",
                        current_price=f"{current_price:.3f}",
                        price_move=f"{price_move_pct:.1%}",
                    )
                    return False, "stale_price"

                # Update order price to current market if it's close
                # This helps orders fill instead of sitting in book
                if price_diff > 0.005:  # More than 0.5 cents different
                    logger.info(
                        "Adjusting order price to current market",
                        outcome=opp.bucket.outcome,
                        old_price=f"{order_price:.3f}",
                        new_price=f"{current_price:.3f}",
                    )
                    opp.price = Decimal(str(round(current_price, 3)))

        # Build trade request with capped size
        request = TradeRequest(
            token_id=opp.bucket.token_id,
            condition_id=opp.market.condition_id,
            outcome=opp.bucket.outcome,
            side=opp.side,
            price=opp.price,
            size=capped_size,
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

        # Update prediction record to mark as traded (for ML training)
        if opp.market.target_date:
            pred = await self.datastore.get_prediction_by_token(
                token_id=opp.bucket.token_id,
                target_date=opp.market.target_date,
            )
            if pred:
                await self.datastore.update_prediction_traded(
                    prediction_id=pred['id'],
                    trade_side=opp.side,
                    trade_price=float(opp.price),
                    trade_size=float(opp.suggested_size),
                )

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
                # Execute the order via Polymarket client
                order_side = OrderSide.BUY if opp.side == "BUY" else OrderSide.SELL
                result = await self.poly_client.place_order(
                    token_id=opp.bucket.token_id,
                    side=order_side,
                    price=opp.price,
                    size=opp.suggested_size,
                )

                if result.success:
                    logger.info(
                        "[SEMI] Trade confirmed and executed",
                        outcome=opp.bucket.outcome,
                        side=opp.side,
                        price=str(opp.price),
                        size=str(opp.suggested_size),
                        order_id=result.order_id,
                    )
                    return True
                else:
                    logger.error(
                        "[SEMI] Trade execution failed",
                        outcome=opp.bucket.outcome,
                        error=result.message,
                    )
                    return False
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
        # Convert dollars to shares (Polymarket orders are in shares, not dollars)
        # suggested_size is in dollars, we need shares = dollars / price
        if opp.price <= 0:
            logger.warning("Invalid price for order", price=str(opp.price))
            return False

        shares = opp.suggested_size / opp.price

        # Polymarket requires minimum $1 order value AND minimum 5 shares
        order_value = shares * opp.price
        MIN_SHARES = Decimal("5")
        if order_value < Decimal("1"):
            logger.info(
                "[AUTO] Order value below $1 minimum, skipping",
                outcome=opp.bucket.outcome,
                order_value=str(order_value),
            )
            return False

        if shares < MIN_SHARES:
            logger.info(
                "[AUTO] Share count below 5 minimum, skipping",
                outcome=opp.bucket.outcome,
                shares=str(shares),
            )
            return False

        # Determine the token to trade (YES token or NO token)
        token_to_trade = opp.effective_token_id or opp.bucket.token_id
        trade_type = "BUY_NO" if opp.is_buy_no else "BUY_YES"

        logger.info(
            "[AUTO] Executing trade",
            outcome=opp.bucket.outcome,
            trade_type=trade_type,
            price=str(opp.price),
            size_dollars=str(opp.suggested_size),
            size_shares=str(shares),
        )

        # Execute the order via Polymarket client
        order_side = OrderSide.BUY if opp.side == "BUY" else OrderSide.SELL
        result = await self.poly_client.place_order(
            token_id=token_to_trade,
            side=order_side,
            price=opp.price,
            size=shares,
        )

        if result.success:
            logger.info(
                "[AUTO] Trade executed",
                city=opp.market.city.name if opp.market.city else "unknown",
                outcome=opp.bucket.outcome,
                trade_type=trade_type,
                price=str(opp.price),
                size=str(opp.suggested_size),
                edge=opp.edge,
                order_id=result.order_id,
            )

            # Record the position so we don't repeat the same trade
            position = Position(
                token_id=token_to_trade,
                condition_id=opp.market.condition_id,
                outcome=opp.bucket.outcome,
                city=opp.market.city.name if opp.market.city else "unknown",
                target_date=opp.market.target_date,
                size=shares,
                entry_price=opp.price,
                current_price=opp.price,
                opened_at=datetime.utcnow(),
            )
            self.risk_manager.record_trade_open(position)

            # Update prediction record to mark as traded (for ML training)
            if opp.market.target_date:
                pred = await self.datastore.get_prediction_by_token(
                    token_id=opp.bucket.token_id,
                    target_date=opp.market.target_date,
                )
                if pred:
                    await self.datastore.update_prediction_traded(
                        prediction_id=pred['id'],
                        trade_side=trade_type,
                        trade_price=float(opp.price),
                        trade_size=float(opp.suggested_size),
                    )

            return True
        else:
            logger.error(
                "[AUTO] Trade execution failed",
                outcome=opp.bucket.outcome,
                error=result.message,
            )
            return False

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


async def session_new(name: str, bankroll: float, notes: str = "") -> None:
    """Start a new trading session."""
    settings = get_settings()
    manager = get_session_manager()

    # Close current session if active
    if manager.current_session:
        print(f"Closing current session: {manager.current_session.name}")
        manager.close_current_session(
            notes="Closed to start new session"
        )

    # Start new session with current settings
    session = manager.start_new_session(
        name=name,
        starting_bankroll=bankroll,
        min_edge_threshold=settings.min_edge_threshold,
        min_confidence=settings.min_confidence,
        max_positions=settings.max_total_positions,
        notes=notes,
    )

    print("\n" + "=" * 60)
    print("🆕 NEW TRADING SESSION STARTED")
    print("=" * 60)
    print(f"  Session ID: {session.session_id}")
    print(f"  Name: {session.name}")
    print(f"  Started: {session.started_at.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Bankroll: ${session.starting_bankroll:.2f}")
    print(f"\n  Settings:")
    print(f"    Min Edge: {session.min_edge_threshold:.0%}")
    print(f"    Min Confidence: {session.min_confidence:.0%}")
    print(f"    Max Positions: {session.max_positions}")
    if notes:
        print(f"\n  Notes: {notes}")
    print("=" * 60)
    print("\n✅ Old positions are now archived. Starting fresh!")


async def session_close(
    total_trades: int = 0,
    winning_trades: int = 0,
    total_pnl: float = 0.0,
    total_cost: float = 0.0,
    notes: str = "",
) -> None:
    """Close the current trading session."""
    manager = get_session_manager()

    if not manager.current_session:
        print("❌ No active session to close.")
        return

    session = manager.close_current_session(
        total_trades=total_trades,
        winning_trades=winning_trades,
        total_pnl=total_pnl,
        total_cost=total_cost,
        notes=notes,
    )

    print("\n" + "=" * 60)
    print("📦 SESSION ARCHIVED")
    print("=" * 60)
    print(f"  Session: {session.name} ({session.session_id})")
    print(f"  Duration: {session.duration_days:.1f} days")
    print(f"  Trades: {session.total_trades}")
    print(f"  Win Rate: {session.win_rate:.1%}")
    print(f"  P&L: ${session.total_pnl:+.2f} ({session.return_pct:.1%})")
    print("=" * 60)


async def session_list() -> None:
    """List all trading sessions."""
    manager = get_session_manager()

    print(manager.compare_sessions())


async def session_status() -> None:
    """Show current session status."""
    manager = get_session_manager()

    if not manager.current_session:
        print("\n❌ No active session.")
        print("   Start one with: python -m src.main session new 'Session Name' --bankroll 100")
        return

    session = manager.current_session
    print("\n" + "=" * 60)
    print("🟢 CURRENT SESSION")
    print("=" * 60)
    print(f"  Session ID: {session.session_id}")
    print(f"  Name: {session.name}")
    print(f"  Started: {session.started_at.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Duration: {session.duration_days:.1f} days")
    print(f"\n  Settings:")
    print(f"    Min Edge: {session.min_edge_threshold:.0%}")
    print(f"    Min Confidence: {session.min_confidence:.0%}")
    print(f"    Max Positions: {session.max_positions}")
    print(f"\n  Performance:")
    print(f"    Trades: {session.total_trades}")
    print(f"    Win Rate: {session.win_rate:.1%}")
    print(f"    P&L: ${session.total_pnl:+.2f} ({session.return_pct:.1%})")
    if session.notes:
        print(f"\n  Notes: {session.notes}")
    print("=" * 60)


async def show_status() -> None:
    """Show current risk and position status."""
    setup_logging()

    risk = RiskManager(
        starting_bankroll=Decimal(str(get_settings().starting_bankroll))
    )
    status = risk.get_status()

    # Also show session status
    manager = get_session_manager()
    if manager.current_session:
        print("\n" + "=" * 60)
        print(f"📊 Current Session: {manager.current_session.name}")
        print(f"   Started: {manager.current_session.started_at.strftime('%Y-%m-%d')}")
        print("=" * 60)

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

    # Session command
    session_parser = subparsers.add_parser(
        "session",
        help="Manage trading sessions",
    )
    session_subparsers = session_parser.add_subparsers(
        dest="session_command",
        help="Session command",
    )

    # session new
    session_new_parser = session_subparsers.add_parser(
        "new",
        help="Start a new trading session (archives old positions)",
    )
    session_new_parser.add_argument(
        "name",
        help="Name for the session (e.g., 'v2-tighter-params')",
    )
    session_new_parser.add_argument(
        "--bankroll", "-b",
        type=float,
        default=100.0,
        help="Starting bankroll for this session (default: 100)",
    )
    session_new_parser.add_argument(
        "--notes", "-n",
        default="",
        help="Notes about this session",
    )

    # session close
    session_close_parser = session_subparsers.add_parser(
        "close",
        help="Close the current session and archive it",
    )
    session_close_parser.add_argument(
        "--trades", "-t",
        type=int,
        default=0,
        help="Total trades in session",
    )
    session_close_parser.add_argument(
        "--wins", "-w",
        type=int,
        default=0,
        help="Winning trades",
    )
    session_close_parser.add_argument(
        "--pnl",
        type=float,
        default=0.0,
        help="Total P&L",
    )
    session_close_parser.add_argument(
        "--cost",
        type=float,
        default=0.0,
        help="Total cost basis",
    )
    session_close_parser.add_argument(
        "--notes", "-n",
        default="",
        help="Notes about session close",
    )

    # session list
    session_subparsers.add_parser(
        "list",
        help="List all trading sessions",
    )

    # session status (default)
    session_subparsers.add_parser(
        "status",
        help="Show current session status",
    )

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
    elif args.command == "session":
        if args.session_command == "new":
            asyncio.run(session_new(args.name, args.bankroll, args.notes))
        elif args.session_command == "close":
            asyncio.run(session_close(
                args.trades, args.wins, args.pnl, args.cost, args.notes
            ))
        elif args.session_command == "list":
            asyncio.run(session_list())
        else:
            asyncio.run(session_status())
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
