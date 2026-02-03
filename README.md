# Polymarket Alpha Bot

A comprehensive signal ingestion and trading system for Polymarket prediction markets.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SIGNAL INGESTION LAYER                            │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
│   Twitter   │    News     │  On-Chain   │  Telegram   │    RSS/Official     │
│   Firehose  │    APIs     │  Monitors   │  Scrapers   │       Feeds         │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────────┬──────────┘
       │             │             │             │                 │
       └─────────────┴─────────────┴─────────────┴─────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SIGNAL PROCESSING LAYER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Signal Parser   │→ │ LLM Interpreter │→ │ Market Matcher              │  │
│  │ (Normalize)     │  │ (Extract Intent)│  │ (Find Relevant Markets)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DECISION ENGINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Probability     │→ │ Edge Calculator │→ │ Position Sizer              │  │
│  │ Estimator       │  │ (EV Analysis)   │  │ (Kelly Criterion)           │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Order Builder   │→ │ Risk Checks     │→ │ Polymarket Executor         │  │
│  │                 │  │ (Limits/Blocks) │  │ (On-Chain Settlement)       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Signal Ingestion (`src/ingestion/`)
- `twitter_monitor.py` - Real-time Twitter/X monitoring via API
- `news_monitor.py` - NewsAPI, Reuters, AP feeds
- `onchain_monitor.py` - Whale wallet tracking, token movements
- `telegram_monitor.py` - Crypto alpha channel monitoring
- `rss_monitor.py` - Official government/institutional feeds

### 2. Signal Processing (`src/processing/`)
- `signal_parser.py` - Normalize signals into standard format
- `llm_interpreter.py` - Use Claude/GPT to extract meaning and implications
- `market_matcher.py` - Map signals to relevant Polymarket markets

### 3. Market Management (`src/markets/`)
- `polymarket_client.py` - API client for Polymarket
- `market_cache.py` - Local cache of active markets with metadata
- `price_monitor.py` - Real-time price tracking

### 4. Execution (`src/execution/`)
- `decision_engine.py` - Core trading logic
- `position_sizer.py` - Kelly criterion and risk management
- `executor.py` - Order placement and management

### 5. Utilities (`src/utils/`)
- `database.py` - SQLite for signal/trade logging
- `logger.py` - Structured logging
- `config.py` - Configuration management

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `config/config.example.yaml` to `config/config.yaml`
3. Add your API keys
4. Run `python main.py`

## Configuration

See `config/config.example.yaml` for all options.

## Risk Warning

This is experimental software. Prediction markets carry significant financial risk.
Only trade with money you can afford to lose.
