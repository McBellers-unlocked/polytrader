#!/usr/bin/env python3
"""
Example: Signal Processing Pipeline

This script demonstrates the complete signal processing pipeline
from ingestion to trade decision.

Run with: python examples/signal_pipeline.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.base import RawSignal
from src.processing.signal_prioritizer import (
    SignalPrioritizer,
    SignalQueue,
    SourceReliabilityTracker,
)
from src.processing.llm_interpreter import MarketMatcher
from src.markets.polymarket_client import PolymarketClient


# Example signals that might trigger trades
EXAMPLE_SIGNALS = [
    {
        "source": "twitter",
        "content": "BREAKING: Federal Reserve announces emergency 50 basis point rate cut, citing banking sector concerns",
        "metadata": {"username": "Reuters"},
    },
    {
        "source": "twitter",
        "content": "JUST IN: Kevin McCarthy elected Speaker of the House on 15th ballot",
        "metadata": {"username": "AP"},
    },
    {
        "source": "news",
        "content": "Bitcoin surges past $100,000 for the first time as institutional adoption accelerates",
        "metadata": {"source_name": "Bloomberg"},
    },
    {
        "source": "twitter",
        "content": "Sources: Lakers finalizing trade to acquire Giannis Antetokounmpo from Milwaukee",
        "metadata": {"username": "wojespn"},
    },
    {
        "source": "rss",
        "content": "SEC approves first spot Ethereum ETF applications from major asset managers",
        "metadata": {"feed_url": "https://sec.gov/news/rss"},
    },
    {
        "source": "onchain",
        "content": "Whale Alert: 50,000 BTC ($5B) transferred from unknown wallet to Coinbase",
        "metadata": {},
    },
    {
        "source": "twitter",
        "content": "Just had coffee ☕",
        "metadata": {"username": "random_user"},
    },
]


async def main():
    print("=" * 70)
    print("POLYMARKET ALPHA BOT - Signal Processing Demo")
    print("=" * 70)
    
    # Initialize components
    reliability_tracker = SourceReliabilityTracker()
    prioritizer = SignalPrioritizer(reliability_tracker)
    signal_queue = SignalQueue(prioritizer)
    market_matcher = MarketMatcher()
    
    # Fetch real markets from Polymarket
    print("\n📊 Fetching active markets from Polymarket...")
    polymarket = PolymarketClient()
    
    try:
        markets = await polymarket.get_markets(limit=200)
        print(f"   Loaded {len(markets)} active markets")
        
        # Convert to format for matching
        markets_for_matching = polymarket.format_markets_for_llm(markets)
        
    except Exception as e:
        print(f"   ⚠️ Could not fetch markets: {e}")
        print("   Using mock market data for demo...")
        markets_for_matching = [
            {"market_id": "fed_rate", "question": "Will the Fed cut rates by 50+ bps in 2024?", "yes_price": 0.25, "category": "economics"},
            {"market_id": "btc_100k", "question": "Will Bitcoin reach $100,000 by end of 2024?", "yes_price": 0.45, "category": "crypto"},
            {"market_id": "eth_etf", "question": "Will an Ethereum ETF be approved in 2024?", "yes_price": 0.60, "category": "crypto"},
            {"market_id": "speaker", "question": "Will Kevin McCarthy be Speaker through 2024?", "yes_price": 0.70, "category": "politics"},
        ]
    
    print("\n" + "-" * 70)
    print("PROCESSING SIGNALS")
    print("-" * 70)
    
    for i, signal_data in enumerate(EXAMPLE_SIGNALS, 1):
        # Create signal
        signal = RawSignal(
            source=signal_data["source"],
            signal_type="test",
            content=signal_data["content"],
            timestamp=datetime.now(timezone.utc),
            metadata=signal_data["metadata"],
        )
        
        print(f"\n🔔 Signal {i}: [{signal.source.upper()}]")
        print(f"   {signal.content[:70]}...")
        
        # Score and queue the signal
        scored = signal_queue.add(signal, markets_for_matching)
        
        if scored:
            print(f"\n   📈 SCORES:")
            print(f"      Priority:    {scored.priority_score:5.1f}")
            print(f"      Urgency:     {scored.urgency_score:5.1f}")
            print(f"      Reliability: {scored.reliability_score:5.1f}")
            print(f"      Relevance:   {scored.relevance_score:5.1f}")
            print(f"      TOTAL:       {scored.total_score:5.1f}")
            
            # Find matching markets
            matches = market_matcher.find_potential_matches(
                signal.content,
                markets_for_matching,
                max_matches=3,
            )
            
            if matches:
                print(f"\n   🎯 POTENTIAL MARKETS:")
                for match in matches:
                    print(f"      • {match['question'][:50]}...")
                    print(f"        Current YES price: ${match['yes_price']:.2f}")
            else:
                print(f"\n   ❌ No matching markets found")
        else:
            print(f"   ⏭️ Filtered out (low priority or duplicate)")
    
    # Show queue stats
    print("\n" + "-" * 70)
    print("QUEUE STATUS")
    print("-" * 70)
    
    stats = signal_queue.get_stats()
    print(f"\nSignals in queue: {stats['size']}")
    print(f"Average score: {stats['avg_score']:.1f}")
    print(f"Highest score: {stats['max_score']:.1f}")
    print(f"Lowest score: {stats['min_score']:.1f}")
    
    # Process queue
    print("\n" + "-" * 70)
    print("PROCESSING QUEUE (highest priority first)")
    print("-" * 70)
    
    position = 1
    while len(signal_queue) > 0:
        scored = signal_queue.pop()
        print(f"\n{position}. Score {scored.total_score:.1f}: {scored.signal.content[:60]}...")
        position += 1
    
    await polymarket.close()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("""
Next steps to make this production-ready:

1. Add your API keys to config/config.yaml:
   - Twitter Bearer Token
   - Anthropic/OpenAI API key
   - Polymarket credentials (for trading)

2. Run the CLI to test:
   python cli.py test-signal "Your test signal here" --analyze

3. Run a backtest:
   python cli.py backtest --days 30

4. Start the bot:
   python main.py --paper  # Paper trading mode first!

5. Monitor via dashboard:
   python cli.py monitor
   Open http://localhost:8000
""")


if __name__ == "__main__":
    asyncio.run(main())
