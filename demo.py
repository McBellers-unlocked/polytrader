#!/usr/bin/env python3
"""
Demo Script - Shows the bot processing signals

This script simulates the bot's behavior without requiring API keys,
demonstrating how signals flow through the system.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logging, get_logger
from src.ingestion.base import RawSignal
from src.processing.signal_processor import (
    SignalProcessor, EntityExtractor, SignalClassifier
)
from src.processing.llm_interpreter import MarketMatcher


# Sample signals for demonstration
DEMO_SIGNALS = [
    RawSignal(
        source="twitter",
        signal_type="tweet",
        content="BREAKING: Fed Chair Powell announces surprise 50bp rate cut, citing deteriorating economic conditions",
        timestamp=datetime.now(timezone.utc),
        metadata={"username": "WSJ", "verified": True},
    ),
    RawSignal(
        source="twitter",
        signal_type="tweet",
        content="🚨 Woj: LeBron James has requested a trade from the Lakers, per sources",
        timestamp=datetime.now(timezone.utc),
        metadata={"username": "wojespn", "verified": True},
    ),
    RawSignal(
        source="news",
        signal_type="article",
        content="Bitcoin ETF sees record $2.1B inflows as institutional interest surges following SEC approval",
        timestamp=datetime.now(timezone.utc),
        metadata={"source_name": "Bloomberg"},
    ),
    RawSignal(
        source="onchain",
        signal_type="whale_transfer",
        content="Jump Trading moved 50,000,000 USDC from Binance to unknown wallet",
        timestamp=datetime.now(timezone.utc),
        metadata={"wallet_label": "Jump Trading", "value_usd": 50000000},
    ),
    RawSignal(
        source="rss",
        signal_type="feed_entry",
        content="OFFICIAL: President Biden signs executive order implementing new AI safety regulations",
        timestamp=datetime.now(timezone.utc),
        metadata={"feed_title": "White House", "category": "politics"},
    ),
    RawSignal(
        source="twitter",
        signal_type="tweet",
        content="Just had a great meeting with investors. Exciting things coming for Tesla in 2025! 🚀",
        timestamp=datetime.now(timezone.utc),
        metadata={"username": "elonmusk", "verified": True},
    ),
    RawSignal(
        source="news",
        signal_type="article",
        content="Ukraine and Russia agree to 72-hour ceasefire, peace talks to resume Monday",
        timestamp=datetime.now(timezone.utc),
        metadata={"source_name": "Reuters"},
    ),
]

# Sample markets for matching
DEMO_MARKETS = [
    {
        "market_id": "fed_rate_cut",
        "question": "Will the Fed cut rates by 50bp or more in January 2025?",
        "yes_price": 0.15,
        "category": "finance",
        "keywords": ["fed", "rate", "cut", "powell"],
        "entities": ["Powell", "Federal Reserve"],
    },
    {
        "market_id": "lebron_trade",
        "question": "Will LeBron James be traded before the 2025 trade deadline?",
        "yes_price": 0.08,
        "category": "sports",
        "keywords": ["lebron", "trade", "lakers", "nba"],
        "entities": ["LeBron James", "Lakers"],
    },
    {
        "market_id": "btc_100k",
        "question": "Will Bitcoin reach $100,000 by March 2025?",
        "yes_price": 0.45,
        "category": "crypto",
        "keywords": ["bitcoin", "btc", "etf", "crypto"],
        "entities": ["Bitcoin", "SEC"],
    },
    {
        "market_id": "ukraine_ceasefire",
        "question": "Will there be a Russia-Ukraine ceasefire lasting 30+ days in 2025?",
        "yes_price": 0.22,
        "category": "geopolitics",
        "keywords": ["ukraine", "russia", "ceasefire", "peace"],
        "entities": ["Ukraine", "Russia"],
    },
    {
        "market_id": "ai_regulation",
        "question": "Will Biden sign AI regulation executive order by Feb 2025?",
        "yes_price": 0.65,
        "category": "politics",
        "keywords": ["ai", "regulation", "biden", "executive"],
        "entities": ["Biden", "AI"],
    },
]


async def demo_signal_processing():
    """Demonstrate signal processing pipeline"""
    print("\n" + "="*70)
    print("📡 SIGNAL PROCESSING DEMO")
    print("="*70)
    
    processor = SignalProcessor()
    entity_extractor = EntityExtractor()
    classifier = SignalClassifier()
    market_matcher = MarketMatcher()
    
    for i, signal in enumerate(DEMO_SIGNALS, 1):
        print(f"\n{'─'*70}")
        print(f"Signal #{i}: [{signal.source.upper()}]")
        print(f"{'─'*70}")
        print(f"📝 Content: {signal.content[:100]}...")
        
        # Process signal
        processed = await processor.process(signal)
        
        if processed:
            print(f"\n🔍 Analysis:")
            print(f"   Categories: {', '.join(processed.categories)}")
            print(f"   Urgency: {'🔴 HIGH' if processed.urgency_score > 0.7 else '🟡 MEDIUM' if processed.urgency_score > 0.4 else '🟢 LOW'} ({processed.urgency_score:.2f})")
            print(f"   Confidence: {processed.confidence_score:.2f}")
            print(f"   Entities: {', '.join(processed.entities[:5]) if processed.entities else 'None detected'}")
            
            # Match to markets
            potential_markets = market_matcher.find_potential_matches(
                signal.content,
                DEMO_MARKETS,
            )
            
            if potential_markets:
                print(f"\n🎯 Potential Market Matches:")
                for market in potential_markets[:3]:
                    print(f"   • {market['question'][:50]}...")
                    print(f"     Current: {market['yes_price']:.2f} | Category: {market['category']}")
            else:
                print(f"\n   No market matches found")
        else:
            print(f"\n⚠️ Signal filtered out (duplicate or low quality)")
        
        await asyncio.sleep(0.1)  # Small delay for readability


async def demo_entity_extraction():
    """Demonstrate entity extraction"""
    print("\n" + "="*70)
    print("🏷️ ENTITY EXTRACTION DEMO")
    print("="*70)
    
    extractor = EntityExtractor()
    
    test_texts = [
        "President Biden met with Fed Chair Powell to discuss inflation concerns.",
        "Tesla CEO Elon Musk announced $10 billion investment in AI infrastructure.",
        "The Lakers are considering trading LeBron James to the Warriors.",
        "$BTC pumped 15% after BlackRock's Bitcoin ETF approval.",
    ]
    
    for text in test_texts:
        print(f"\n📝 Text: {text}")
        entities = extractor.extract(text)
        print(f"🏷️ Entities:")
        for category, items in entities.items():
            if items:
                print(f"   {category}: {', '.join(items)}")


async def demo_market_matching():
    """Demonstrate market matching"""
    print("\n" + "="*70)
    print("🎯 MARKET MATCHING DEMO")
    print("="*70)
    
    matcher = MarketMatcher()
    
    test_signals = [
        "Fed announces emergency rate cut of 75 basis points",
        "LeBron James hints at retirement on Instagram",
        "Bitcoin ETF volume hits all-time high",
        "Unknown signal about weather in Tokyo",
    ]
    
    for signal_text in test_signals:
        print(f"\n📡 Signal: {signal_text}")
        
        matches = matcher.find_potential_matches(signal_text, DEMO_MARKETS)
        
        if matches:
            print(f"✅ Matched {len(matches)} market(s):")
            for m in matches:
                print(f"   • {m['question'][:50]}... (price: {m['yes_price']:.2f})")
        else:
            print(f"❌ No market matches")


async def demo_trade_decision():
    """Demonstrate trade decision logic"""
    print("\n" + "="*70)
    print("💰 TRADE DECISION DEMO")
    print("="*70)
    
    # Simulated scenarios
    scenarios = [
        {
            "signal": "BREAKING: Fed cuts rates 50bp",
            "market": "Fed 50bp cut in January?",
            "current_price": 0.15,
            "estimated_prob": 0.85,
            "confidence": 0.90,
        },
        {
            "signal": "Sources say LeBron considering trade request",
            "market": "LeBron traded by deadline?",
            "current_price": 0.08,
            "estimated_prob": 0.35,
            "confidence": 0.60,
        },
        {
            "signal": "Minor crypto news",
            "market": "BTC to $100k?",
            "current_price": 0.45,
            "estimated_prob": 0.48,
            "confidence": 0.40,
        },
    ]
    
    for scenario in scenarios:
        print(f"\n{'─'*70}")
        print(f"📡 Signal: {scenario['signal']}")
        print(f"📊 Market: {scenario['market']}")
        print(f"💵 Current Price: {scenario['current_price']:.2f}")
        print(f"📈 Estimated Prob: {scenario['estimated_prob']:.2f}")
        print(f"🎯 Confidence: {scenario['confidence']:.2f}")
        
        edge = scenario['estimated_prob'] - scenario['current_price']
        edge_pct = edge * 100
        
        print(f"\n📐 Calculated Edge: {edge_pct:.1f}%")
        
        # Decision logic
        min_edge = 0.15  # 15% minimum edge
        min_confidence = 0.50
        
        should_trade = edge >= min_edge and scenario['confidence'] >= min_confidence
        
        if should_trade:
            # Kelly sizing (simplified)
            kelly = edge / (1 - scenario['current_price'])
            position_pct = min(kelly * 0.25, 0.10)  # Quarter Kelly, max 10%
            
            print(f"✅ TRADE: BUY YES")
            print(f"   Position Size: {position_pct*100:.1f}% of bankroll")
            print(f"   Expected Value: +{edge_pct:.1f}%")
        else:
            reasons = []
            if edge < min_edge:
                reasons.append(f"edge {edge_pct:.1f}% < {min_edge*100:.0f}% minimum")
            if scenario['confidence'] < min_confidence:
                reasons.append(f"confidence {scenario['confidence']:.2f} < {min_confidence} minimum")
            
            print(f"❌ NO TRADE: {', '.join(reasons)}")


async def main():
    """Run all demos"""
    setup_logging(level="WARNING")  # Suppress detailed logs for demo
    
    print("\n" + "🤖"*35)
    print("\n     POLYMARKET ALPHA BOT - DEMO")
    print("\n" + "🤖"*35)
    
    print("""
This demo shows how the bot processes signals:
1. Signal Processing - Analyze and categorize incoming signals
2. Entity Extraction - Identify people, companies, tokens mentioned
3. Market Matching - Find relevant prediction markets
4. Trade Decision - Calculate edge and decide whether to trade
""")
    
    await demo_signal_processing()
    await demo_entity_extraction()
    await demo_market_matching()
    await demo_trade_decision()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("""
To run the full bot:
1. Copy config/config.example.yaml to config/config.yaml
2. Add your API keys (Twitter, NewsAPI, Anthropic, Polymarket)
3. Run: python main.py --paper

The bot will:
• Monitor all configured signal sources
• Process signals through the LLM interpreter
• Match signals to Polymarket markets
• Calculate edge and execute trades (paper mode by default)
""")


if __name__ == "__main__":
    asyncio.run(main())
