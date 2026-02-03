#!/usr/bin/env python3
"""
Polymarket Alpha Bot CLI

Command-line interface for testing and managing the bot.

Usage:
    python cli.py test-signal "Breaking news: Fed announces rate cut"
    python cli.py fetch-markets
    python cli.py backtest --days 30
    python cli.py monitor
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config, override_from_env
from src.utils.logger import setup_logging, get_logger
from src.utils.database import Database, Signal

from src.ingestion.base import RawSignal
from src.processing.llm_interpreter import LLMInterpreter, MarketMatcher
from src.processing.signal_prioritizer import SignalPrioritizer, SourceReliabilityTracker

from src.markets.polymarket_client import PolymarketClient

from src.execution.backtesting import (
    BacktestEngine,
    HistoricalDataLoader,
    example_strategy,
)


async def cmd_test_signal(args):
    """Test how a signal would be processed"""
    config = load_config(args.config)
    config = override_from_env(config)
    
    logger = get_logger("cli")
    logger.info("Testing signal processing")
    
    # Create signal
    signal = RawSignal(
        source=args.source,
        signal_type="test",
        content=args.content,
        timestamp=datetime.now(timezone.utc),
        metadata={"test": True},
    )
    
    # Score signal
    prioritizer = SignalPrioritizer()
    scored = prioritizer.score_signal(signal)
    
    print("\n" + "=" * 60)
    print("SIGNAL ANALYSIS")
    print("=" * 60)
    print(f"\nContent: {signal.content}")
    print(f"Source: {signal.source}")
    print(f"\nPriority Score: {scored.priority_score:.1f}")
    print(f"Urgency Score: {scored.urgency_score:.1f}")
    print(f"Reliability Score: {scored.reliability_score:.1f}")
    print(f"Relevance Score: {scored.relevance_score:.1f}")
    print(f"Total Score: {scored.total_score:.1f}")
    
    print("\nScore Components:")
    for name, value in scored.components.items():
        print(f"  {name}: {value:.1f}")
    
    # If LLM credentials available, do full analysis
    if config.credentials.llm.anthropic_api_key or config.credentials.llm.openai_api_key:
        print("\n" + "-" * 60)
        print("LLM ANALYSIS")
        print("-" * 60)
        
        # Initialize Polymarket client
        polymarket = PolymarketClient()
        markets = await polymarket.get_markets(limit=100)
        
        print(f"\nFetched {len(markets)} active markets")
        
        # Pre-filter markets
        matcher = MarketMatcher()
        potential_markets = matcher.find_potential_matches(
            signal.content,
            polymarket.format_markets_for_llm(markets),
            max_matches=20,
        )
        
        print(f"Found {len(potential_markets)} potentially relevant markets")
        
        if potential_markets:
            print("\nTop potential matches:")
            for i, market in enumerate(potential_markets[:5]):
                print(f"  {i+1}. {market['question'][:60]}...")
                print(f"     Price: YES ${market['yes_price']:.2f}")
        
        # LLM analysis
        if args.analyze:
            llm = LLMInterpreter(
                credentials=config.credentials.llm,
                temperature=0.1,
            )
            
            print("\nRunning LLM analysis...")
            analysis = await llm.analyze_signal(
                signal_id=signal.id,
                signal_content=signal.content,
                signal_source=signal.source,
                signal_metadata=signal.metadata,
                available_markets=potential_markets,
            )
            
            print(f"\nActionable: {analysis.is_actionable}")
            print(f"Summary: {analysis.summary}")
            print(f"Entities: {', '.join(analysis.entities)}")
            print(f"Categories: {', '.join(analysis.categories)}")
            print(f"Analysis Time: {analysis.analysis_time_ms:.0f}ms")
            
            if analysis.market_matches:
                print("\nMarket Matches:")
                for match in analysis.market_matches:
                    print(f"\n  Market: {match.market_question[:50]}...")
                    print(f"  Current Price: {match.current_price:.2f}")
                    print(f"  Estimated Prob: {match.estimated_probability:.2f}")
                    print(f"  Edge: {match.edge * 100:.1f}%")
                    print(f"  Confidence: {match.confidence:.2f}")
                    print(f"  Side: {match.recommended_side.upper()}")
                    print(f"  Reasoning: {match.reasoning[:100]}...")
        
        await polymarket.close()
    else:
        print("\nNote: Set LLM API keys for full analysis")


async def cmd_fetch_markets(args):
    """Fetch and display active markets"""
    config = load_config(args.config)
    
    logger = get_logger("cli")
    logger.info("Fetching markets from Polymarket")
    
    polymarket = PolymarketClient()
    markets = await polymarket.get_markets(limit=args.limit)
    
    print(f"\nFetched {len(markets)} markets")
    print("=" * 80)
    
    # Filter by category if specified
    if args.category:
        markets = [m for m in markets if args.category.lower() in m.category.lower()]
        print(f"Filtered to {len(markets)} markets in category '{args.category}'")
    
    # Sort by liquidity
    markets.sort(key=lambda m: m.liquidity, reverse=True)
    
    for i, market in enumerate(markets[:args.show]):
        print(f"\n{i+1}. {market.question}")
        print(f"   ID: {market.id}")
        print(f"   YES: ${market.yes_price:.2f}  NO: ${market.no_price:.2f}")
        print(f"   Liquidity: ${market.liquidity:,.0f}  Volume 24h: ${market.volume_24h:,.0f}")
        print(f"   Category: {market.category}")
        if market.end_date:
            print(f"   Ends: {market.end_date.strftime('%Y-%m-%d')}")
    
    await polymarket.close()
    
    if args.json:
        output = {
            "markets": [
                {
                    "id": m.id,
                    "question": m.question,
                    "yes_price": m.yes_price,
                    "no_price": m.no_price,
                    "liquidity": m.liquidity,
                    "volume_24h": m.volume_24h,
                    "category": m.category,
                }
                for m in markets
            ]
        }
        
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.json}")


async def cmd_backtest(args):
    """Run a backtest with synthetic data"""
    config = load_config(args.config)
    
    logger = get_logger("cli")
    logger.info("Running backtest")
    
    # Initialize database
    database = Database(config.database.path)
    await database.initialize()
    
    # Generate synthetic data
    loader = HistoricalDataLoader(database)
    
    start_date = datetime.now(timezone.utc) - timedelta(days=args.days)
    end_date = datetime.now(timezone.utc)
    
    print(f"Generating synthetic data for {args.days} days...")
    signals, markets = loader.generate_synthetic_data(
        num_signals=args.signals,
        num_markets=args.markets,
        start_date=start_date,
        end_date=end_date,
    )
    
    print(f"Generated {len(signals)} signals and {len(markets)} markets")
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=args.capital,
        max_position_pct=0.10,
        trading_fee_pct=0.01,
    )
    
    print("Running backtest...")
    result = await engine.run_backtest(
        signals=signals,
        markets=markets,
        strategy=example_strategy,
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\nPeriod: {result.start_time.date()} to {result.end_time.date()}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Final Capital: ${result.metrics.get('final_capital', 0):,.2f}")
    
    print(f"\nTotal Signals: {result.total_signals}")
    print(f"Signals Traded: {result.signals_traded}")
    print(f"Total Trades: {result.total_trades}")
    
    print(f"\nWinning Trades: {result.winning_trades}")
    print(f"Losing Trades: {result.losing_trades}")
    print(f"Win Rate: {result.metrics.get('win_rate', 0) * 100:.1f}%")
    
    print(f"\nTotal P&L: ${result.total_pnl:,.2f}")
    print(f"ROI: {result.roi_percent:.1f}%")
    print(f"Max Drawdown: {result.max_drawdown:.1f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    
    if result.metrics.get('profit_factor'):
        print(f"Profit Factor: {result.metrics['profit_factor']:.2f}")
    
    print(f"\nAvg Win: ${result.metrics.get('avg_win', 0):.2f}")
    print(f"Avg Loss: ${result.metrics.get('avg_loss', 0):.2f}")


async def cmd_monitor(args):
    """Start monitoring server only"""
    config = load_config(args.config)
    config = override_from_env(config)
    
    setup_logging(level="INFO")
    logger = get_logger("cli")
    
    # Initialize components
    database = Database(config.database.path)
    await database.initialize()
    
    from src.utils.monitoring import (
        MetricsCollector,
        DashboardData,
        MonitoringServer,
    )
    
    metrics = MetricsCollector()
    dashboard = DashboardData(database, metrics)
    server = MonitoringServer(metrics, dashboard, port=args.port)
    
    print(f"Starting monitoring server on port {args.port}...")
    print(f"Dashboard: http://localhost:{args.port}/")
    print(f"Metrics: http://localhost:{args.port}/metrics")
    print(f"Health: http://localhost:{args.port}/health")
    print("\nPress Ctrl+C to stop")
    
    await server.start()
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        await server.stop()


async def cmd_search_markets(args):
    """Search markets by keyword"""
    config = load_config(args.config)
    
    polymarket = PolymarketClient()
    markets = await polymarket.search_markets(
        query=args.query,
        min_liquidity=args.min_liquidity,
    )
    
    print(f"\nFound {len(markets)} markets matching '{args.query}'")
    print("=" * 80)
    
    for i, market in enumerate(markets[:20]):
        print(f"\n{i+1}. {market.question}")
        print(f"   YES: ${market.yes_price:.2f}  NO: ${market.no_price:.2f}")
        print(f"   Liquidity: ${market.liquidity:,.0f}")
    
    await polymarket.close()


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Alpha Bot CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # test-signal command
    test_parser = subparsers.add_parser(
        "test-signal",
        help="Test how a signal would be processed",
    )
    test_parser.add_argument("content", help="Signal content to test")
    test_parser.add_argument(
        "--source",
        default="twitter",
        help="Signal source (twitter, news, rss)",
    )
    test_parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run full LLM analysis",
    )
    
    # fetch-markets command
    markets_parser = subparsers.add_parser(
        "fetch-markets",
        help="Fetch active markets from Polymarket",
    )
    markets_parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum markets to fetch",
    )
    markets_parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="Number of markets to display",
    )
    markets_parser.add_argument(
        "--category",
        help="Filter by category",
    )
    markets_parser.add_argument(
        "--json",
        help="Save to JSON file",
    )
    
    # search-markets command
    search_parser = subparsers.add_parser(
        "search-markets",
        help="Search markets by keyword",
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--min-liquidity",
        type=float,
        default=10000,
        help="Minimum liquidity",
    )
    
    # backtest command
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run a backtest",
    )
    backtest_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to simulate",
    )
    backtest_parser.add_argument(
        "--signals",
        type=int,
        default=1000,
        help="Number of synthetic signals",
    )
    backtest_parser.add_argument(
        "--markets",
        type=int,
        default=50,
        help="Number of synthetic markets",
    )
    backtest_parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Initial capital",
    )
    
    # monitor command
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Start monitoring server",
    )
    monitor_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port",
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup basic logging
    setup_logging(level="WARNING")
    
    # Run the appropriate command
    if args.command == "test-signal":
        asyncio.run(cmd_test_signal(args))
    elif args.command == "fetch-markets":
        asyncio.run(cmd_fetch_markets(args))
    elif args.command == "search-markets":
        asyncio.run(cmd_search_markets(args))
    elif args.command == "backtest":
        asyncio.run(cmd_backtest(args))
    elif args.command == "monitor":
        asyncio.run(cmd_monitor(args))


if __name__ == "__main__":
    main()
