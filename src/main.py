"""Main entry point for Polytrader weather trading bot."""

import argparse
import asyncio
import signal
import sys
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any

from src.config import get_settings, TradingMode, CITIES, CityConfig
from src.logging import setup_logging, get_logger
from src.data import WeatherAggregator
from src.markets import MarketScanner, PolymarketClient, WeatherMarket
from src.strategy import FairValueCalculator, EdgeDetector, ArbitrageDetector, TradeSignal
from src.risk import RiskManager
from src.execution import ExecutionEngine

logger = get_logger(__name__)


class TradingBot:
    """
    Main trading bot that orchestrates all components.

    Runs a continuous loop that:
    1. Scans for active weather markets
    2. Fetches ensemble weather forecasts
    3. Calculates fair values and detects edge
    4. Executes trades when criteria are met
    """

    def __init__(self):
        """Initialize the trading bot."""
        self.settings = get_settings()
        self.weather = WeatherAggregator()
        self.scanner = MarketScanner()
        self.client = PolymarketClient()
        self.fair_value_calc = FairValueCalculator()
        self.edge_detector = EdgeDetector()
        self.arb_detector = ArbitrageDetector()
        self.risk_manager = RiskManager()
        self.execution = ExecutionEngine(self.client, self.risk_manager)

        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the trading bot."""
        logger.info(
            "Starting Polytrader",
            mode=self.settings.trading_mode.value,
            bankroll=self.settings.starting_bankroll,
        )

        # Validate settings
        self.settings.validate_live_trading()

        # Start execution engine
        await self.execution.start()

        self._running = True

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

        await self.execution.stop()
        await self.scanner.close()

    def _handle_shutdown(self) -> None:
        """Handle shutdown signal."""
        logger.info("Shutdown signal received")
        self._running = False
        self._shutdown_event.set()

    async def _run_loop(self) -> None:
        """Main trading loop."""
        while self._running:
            try:
                await self._run_iteration()

                # Wait before next iteration
                logger.info("Waiting for next iteration", wait_seconds=60)
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=60,
                    )
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error("Error in trading loop", error=str(e), exc_info=True)
                await asyncio.sleep(30)  # Back off on error

    async def _run_iteration(self) -> None:
        """Run a single trading iteration."""
        logger.info("Starting trading iteration")

        # 1. Scan for weather markets
        markets = await self.scanner.scan_weather_markets()
        if not markets:
            logger.info("No active weather markets found")
            return

        logger.info("Found weather markets", count=len(markets))

        # 2. Process each market
        all_signals: list[TradeSignal] = []
        async with self.weather:
            for market in markets:
                try:
                    signals = await self._process_market(market)
                    all_signals.extend(signals)
                except Exception as e:
                    logger.warning(
                        "Error processing market",
                        market=market.condition_id,
                        error=str(e),
                    )

        # 3. Check for arbitrage opportunities
        arb_opportunities = self.arb_detector.find_all_arbitrage(markets)
        for arb in arb_opportunities:
            logger.info(
                "Arbitrage opportunity found",
                market=arb.market.condition_id,
                profit_margin=arb.profit_margin,
                roi_pct=arb.roi,
            )

        # 4. Execute best signals
        if all_signals:
            # Filter and sort signals
            best_signals = self.edge_detector.filter_signals(all_signals, max_signals=3)

            for signal in best_signals:
                await self._execute_signal(signal)

        # 5. Log status
        status = self.execution.get_status()
        logger.info("Iteration complete", status=status)

    async def _process_market(
        self,
        market: WeatherMarket,
    ) -> list[TradeSignal]:
        """Process a single market for trading signals."""
        if not market.city or not market.target_date:
            return []

        # Skip markets closing in less than 4 hours (too close to resolution)
        if market.hours_until_close < 4:
            logger.debug(
                "Skipping market too close to resolution",
                market=market.condition_id,
                hours_until_close=market.hours_until_close,
            )
            return []

        # Fetch weather forecast
        forecast = await self.weather.get_forecast(
            market.city,
            market.target_date,
        )

        if not forecast.has_sufficient_data:
            logger.warning(
                "Insufficient forecast data",
                market=market.condition_id,
                city=market.city.name,
            )
            return []

        # Calculate fair values
        bucket_probs = self.fair_value_calc.calculate_fair_values(market, forecast)
        bucket_probs = self.fair_value_calc.calibrate_probabilities(bucket_probs)

        # Get current bankroll
        bankroll = Decimal(str(self.settings.starting_bankroll))
        balance = await self.client.get_balance()
        if "USDC" in balance:
            bankroll = balance["USDC"]

        # Detect edge
        signals = self.edge_detector.detect_signals(
            market,
            bucket_probs,
            forecast.model_agreement,
            bankroll,
        )

        if signals:
            logger.info(
                "Signals detected",
                market=market.condition_id,
                city=market.city.name,
                n_signals=len(signals),
                best_edge=signals[0].edge if signals else 0,
            )

        return signals

    async def _execute_signal(self, signal: TradeSignal) -> None:
        """Execute a trade signal."""
        logger.info(
            "Executing signal",
            market=signal.market.condition_id,
            outcome=signal.bucket.outcome,
            edge=signal.edge,
            size=str(signal.suggested_size),
        )

        order = await self.execution.execute_signal(signal)

        if order:
            logger.info(
                "Order created",
                order_id=order.id,
                status=order.status.value,
            )


async def run_bot(mode: str) -> None:
    """Run the trading bot."""
    # Set up logging
    setup_logging()

    # Override mode if specified
    settings = get_settings()
    if mode:
        settings.trading_mode = TradingMode(mode)

    # Create and run bot
    bot = TradingBot()
    await bot.start()


async def run_backtest(
    city: str,
    start_date: str,
    end_date: str,
) -> None:
    """Run a backtest simulation."""
    setup_logging()
    logger.info(
        "Running backtest",
        city=city,
        start_date=start_date,
        end_date=end_date,
    )

    city_config = CITIES.get(city.lower())
    if not city_config:
        logger.error(f"Unknown city: {city}")
        return

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    async with WeatherAggregator() as weather:
        current = start
        while current <= end:
            try:
                forecast = await weather.get_forecast(city_config, current)
                stats = forecast.get_distribution_stats()
                logger.info(
                    "Backtest day",
                    date=str(current),
                    city=city_config.name,
                    mean=stats.get("mean"),
                    std=stats.get("std"),
                    model_agreement=forecast.model_agreement,
                )
            except Exception as e:
                logger.warning(
                    "Backtest error",
                    date=str(current),
                    error=str(e),
                )

            current += timedelta(days=1)


async def show_status() -> None:
    """Show current bot status."""
    setup_logging()

    execution = ExecutionEngine()
    await execution.start()

    status = execution.get_status()

    print("\n" + "=" * 60)
    print("POLYTRADER STATUS")
    print("=" * 60)
    print(f"  Active Orders: {status['active_orders']}")
    print(f"  Open Positions: {status['open_positions']}")
    print(f"  Total Position Value: ${status['total_position_value']}")
    print(f"  Unrealized P&L: ${status['total_unrealized_pnl']}")
    print(f"  Realized P&L: ${status['total_realized_pnl']}")
    print()
    print("Risk Status:")
    risk = status['risk_status']
    print(f"  Current Bankroll: ${risk['current_bankroll']}")
    print(f"  Drawdown: {risk['current_drawdown']:.1%}")
    print(f"  Today P&L: {risk['today_pnl']:.1%}")
    print(f"  Emergency Stop: {'ACTIVE' if risk['is_emergency_stopped'] else 'OK'}")
    print("=" * 60)

    await execution.stop()


async def scan_markets() -> None:
    """Scan and display available weather markets."""
    setup_logging()

    scanner = MarketScanner()
    markets = await scanner.scan_weather_markets()

    print("\n" + "=" * 60)
    print("ACTIVE WEATHER MARKETS")
    print("=" * 60)

    for market in markets:
        print(f"\nMarket: {market.condition_id[:16]}...")
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

    await scanner.close()
    print("=" * 60)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Polytrader - Polymarket weather trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --mode paper     Run in paper trading mode
  python -m src.main --mode semi      Run with manual confirmation
  python -m src.main --mode auto      Run fully automated
  python -m src.main scan             Scan for weather markets
  python -m src.main status           Show current status
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

    # Scan command
    subparsers.add_parser("scan", help="Scan for weather markets")

    # Status command
    subparsers.add_parser("status", help="Show current status")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--city", required=True, help="City to backtest")
    backtest_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")

    # Also support --mode at top level for convenience
    parser.add_argument(
        "--mode", "-m",
        choices=["paper", "semi", "auto"],
        help="Trading mode (shortcut for 'run --mode')",
    )

    args = parser.parse_args()

    # Handle commands
    if args.command == "scan":
        asyncio.run(scan_markets())
    elif args.command == "status":
        asyncio.run(show_status())
    elif args.command == "backtest":
        asyncio.run(run_backtest(args.city, args.start, args.end))
    else:
        # Default to run
        mode = args.mode or "paper"
        asyncio.run(run_bot(mode))


if __name__ == "__main__":
    main()
