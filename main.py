#!/usr/bin/env python3
"""
Polymarket Alpha Bot - Main Entry Point

This bot monitors multiple signal sources and automatically trades
on Polymarket when it detects actionable information.

Usage:
    python main.py                    # Run with default config
    python main.py --config path.yaml # Run with custom config
    python main.py --paper            # Force paper trading mode
    python main.py --dry-run          # Process signals but don't trade
"""

import asyncio
import argparse
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config, override_from_env, Config
from src.utils.logger import setup_logging, get_logger
from src.utils.database import Database

from src.ingestion.base import CompositeMonitor
from src.ingestion.twitter_monitor import TwitterMonitor
from src.ingestion.news_monitor import NewsAPIMonitor, RSSMonitor, BreakingNewsMonitor
from src.ingestion.onchain_monitor import OnChainMonitor, WhaleAlertMonitor

from src.processing.llm_interpreter import LLMInterpreter

from src.markets.polymarket_client import PolymarketClient

from src.execution.decision_engine import DecisionEngine, TradingBot


class PolymarketAlphaBot:
    """
    Main application class
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("main")
        
        # Initialize components
        self.database: Database = None
        self.polymarket: PolymarketClient = None
        self.llm: LLMInterpreter = None
        self.monitors: CompositeMonitor = None
        self.engine: DecisionEngine = None
        self.bot: TradingBot = None
        
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self) -> None:
        """Initialize all components"""
        self.logger.info("initializing_bot")
        
        # Database
        self.database = Database(self.config.database.path)
        await self.database.initialize()
        self.logger.info("database_initialized")
        
        # Polymarket client
        self.polymarket = PolymarketClient(
            api_key=self.config.credentials.polymarket.api_key,
            api_secret=self.config.credentials.polymarket.api_secret,
            passphrase=self.config.credentials.polymarket.passphrase,
            private_key=self.config.credentials.polymarket.private_key,
            database=self.database,
        )
        
        # Fetch initial markets
        markets = await self.polymarket.get_markets()
        self.logger.info("markets_loaded", count=len(markets))
        
        # LLM Interpreter
        self.llm = LLMInterpreter(
            credentials=self.config.credentials.llm,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            cache_ttl_seconds=self.config.llm.cache_identical_signals_seconds,
        )
        
        # Decision engine
        self.engine = DecisionEngine(
            polymarket_client=self.polymarket,
            llm_interpreter=self.llm,
            config=self.config.trading,
            database=self.database,
        )
        
        # Trading bot
        self.bot = TradingBot(
            decision_engine=self.engine,
            database=self.database,
        )
        
        # Signal monitors
        self.monitors = CompositeMonitor()
        await self._setup_monitors()
        
        # Register bot callback
        self.monitors.register_callback(self.bot.on_signal)
        
        self.logger.info("bot_initialized")
    
    async def _setup_monitors(self) -> None:
        """Set up all configured signal monitors"""
        sources = self.config.signal_sources
        
        # Twitter monitor
        if sources.twitter.enabled and self.config.credentials.twitter.bearer_token:
            twitter_monitor = TwitterMonitor(
                bearer_token=self.config.credentials.twitter.bearer_token,
                accounts=sources.twitter.accounts,
                keywords=sources.twitter.keywords,
                check_interval=sources.twitter.check_interval_seconds,
                database=self.database,
            )
            self.monitors.add_monitor(twitter_monitor)
            self.logger.info("twitter_monitor_enabled")
        
        # News API monitor
        if sources.news.enabled and self.config.credentials.news.newsapi_key:
            news_monitor = NewsAPIMonitor(
                api_key=self.config.credentials.news.newsapi_key,
                sources=sources.news.sources,
                check_interval=sources.news.check_interval_seconds,
                database=self.database,
            )
            self.monitors.add_monitor(news_monitor)
            self.logger.info("news_monitor_enabled")
        
        # RSS monitor
        if sources.rss.enabled and sources.rss.feeds:
            rss_monitor = RSSMonitor(
                feeds=[{"url": f.url, "category": f.category} for f in sources.rss.feeds],
                check_interval=sources.rss.check_interval_seconds,
                database=self.database,
            )
            self.monitors.add_monitor(rss_monitor)
            self.logger.info("rss_monitor_enabled")
        
        # Breaking news monitor (always enabled, no API key needed)
        breaking_monitor = BreakingNewsMonitor(
            check_interval=15.0,
            database=self.database,
        )
        self.monitors.add_monitor(breaking_monitor)
        self.logger.info("breaking_news_monitor_enabled")
        
        # On-chain monitor
        if sources.onchain.enabled and self.config.credentials.blockchain.ethereum_rpc:
            onchain_monitor = OnChainMonitor(
                ethereum_rpc=self.config.credentials.blockchain.ethereum_rpc,
                polygon_rpc=self.config.credentials.blockchain.polygon_rpc,
                tracked_wallets=[
                    {"address": w.address, "label": w.label}
                    for w in sources.onchain.tracked_wallets
                ],
                tracked_tokens=[
                    {"address": t.address, "symbol": t.symbol, "threshold": t.threshold}
                    for t in sources.onchain.tracked_tokens
                ],
                check_interval=sources.onchain.check_interval_seconds,
                database=self.database,
            )
            self.monitors.add_monitor(onchain_monitor)
            self.logger.info("onchain_monitor_enabled")
        
        # Whale alert monitor
        if self.config.credentials.twitter.bearer_token:
            whale_monitor = WhaleAlertMonitor(
                twitter_bearer_token=self.config.credentials.twitter.bearer_token,
                min_usd_value=10_000_000,
                check_interval=10.0,
                database=self.database,
            )
            self.monitors.add_monitor(whale_monitor)
            self.logger.info("whale_alert_monitor_enabled")
    
    async def run(self) -> None:
        """Main run loop"""
        self.logger.info("starting_bot")
        
        # Start all components concurrently
        tasks = [
            asyncio.create_task(self.monitors.start_all()),
            asyncio.create_task(self.bot.start()),
            asyncio.create_task(self._status_loop()),
        ]
        
        try:
            # Wait for shutdown signal
            await self._shutdown_event.wait()
        finally:
            # Cleanup
            self.logger.info("shutting_down")
            self.bot.stop()
            self.monitors.stop_all()
            
            for task in tasks:
                task.cancel()
            
            await self.polymarket.close()
    
    async def _status_loop(self) -> None:
        """Periodically log status"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)
                
                stats = await self.bot.get_stats()
                health = await self.monitors.health_check_all()
                
                self.logger.info(
                    "status_update",
                    trades=stats["total_trades"],
                    queue_size=stats["queue_size"],
                    monitors_healthy=sum(health.values()),
                    monitors_total=len(health),
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("status_loop_error", error=str(e))
    
    def shutdown(self) -> None:
        """Trigger shutdown"""
        self._shutdown_event.set()


def setup_signal_handlers(bot: PolymarketAlphaBot) -> None:
    """Set up signal handlers for graceful shutdown"""
    def handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        bot.shutdown()
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


async def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Polymarket Alpha Bot"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Force paper trading mode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process signals but don't trade",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config = override_from_env(config)
    
    # Override trading settings if requested
    if args.paper:
        config.trading.paper_trading = True
    if args.dry_run:
        config.trading.enabled = False
    
    # Setup logging
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
        max_size_mb=config.logging.max_size_mb,
        backup_count=config.logging.backup_count,
    )
    
    logger = get_logger("main")
    logger.info(
        "starting",
        config_path=args.config,
        paper_trading=config.trading.paper_trading,
        trading_enabled=config.trading.enabled,
    )
    
    # Create and run bot
    bot = PolymarketAlphaBot(config)
    setup_signal_handlers(bot)
    
    try:
        await bot.initialize()
        await bot.run()
    except Exception as e:
        logger.error("fatal_error", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
