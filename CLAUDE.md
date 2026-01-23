# CLAUDE.md - AI Assistant Guide for Polytrader

This document provides guidance for AI assistants working with the Polytrader codebase.

## Project Overview

**Polytrader** is a Polymarket weather trading bot that identifies mispriced "Highest temperature in [CITY] on [DATE]" prediction markets by comparing ensemble weather forecasts against market prices, then executes trades when edge > 10%.

### Key Features
- Multi-model ensemble weather forecasting (200+ members)
- Fair value calculation from probability distributions
- Edge detection and Kelly-based position sizing
- Arbitrage detection for rebalancing opportunities
- Risk management with drawdown stops
- Paper/semi-auto/full-auto trading modes

## Repository Structure

```
polytrader/
├── src/
│   ├── __init__.py              # Package init
│   ├── config.py                # Configuration and settings (pydantic-settings)
│   ├── logging.py               # Structured logging (structlog)
│   ├── main.py                  # CLI entry point
│   ├── data/                    # Weather data aggregation
│   │   ├── __init__.py
│   │   ├── ensemble.py          # Open-Meteo ensemble API client
│   │   ├── tomorrow.py          # Tomorrow.io API client
│   │   ├── metar.py             # Aviation weather METAR client
│   │   └── aggregator.py        # Multi-source data aggregation
│   ├── markets/                 # Polymarket integration
│   │   ├── __init__.py
│   │   ├── scanner.py           # Market discovery via Gamma API
│   │   └── client.py            # CLOB API for order execution
│   ├── strategy/                # Trading strategy
│   │   ├── __init__.py
│   │   ├── fair_value.py        # Fair value from ensemble distribution
│   │   ├── edge.py              # Edge detection and signal generation
│   │   └── arbitrage.py         # Rebalancing arbitrage detection
│   ├── risk/                    # Risk management
│   │   ├── __init__.py
│   │   └── manager.py           # Position limits, drawdown stops
│   └── execution/               # Order execution
│       ├── __init__.py
│       ├── engine.py            # Execution orchestration
│       ├── orders.py            # Order data structures
│       ├── positions.py         # Position tracking
│       └── persistence.py       # SQLite storage
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_ensemble.py
│   └── test_strategy.py
├── pyproject.toml               # Project config and dependencies
├── .env.example                 # Environment variable template
├── .gitignore
└── CLAUDE.md                    # This file
```

## Development Setup

### Prerequisites
- Python 3.11+
- pip or uv package manager

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd polytrader

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRADING_MODE` | paper / semi / auto | paper |
| `STARTING_BANKROLL` | Initial bankroll in USD | 1500 |
| `POLYMARKET_PRIVATE_KEY` | Wallet private key (for live trading) | - |
| `POLYMARKET_FUNDER` | Funder address | - |
| `TOMORROW_IO_API_KEY` | Tomorrow.io API key (optional) | - |
| `MIN_EDGE_THRESHOLD` | Minimum edge to trade | 0.10 |
| `MIN_MODEL_AGREEMENT` | Minimum model agreement | 0.65 |
| `MAX_POSITION_PCT` | Max position as % of bankroll | 0.02 |
| `DAILY_LOSS_STOP_PCT` | Daily loss stop | 0.05 |
| `MAX_DRAWDOWN_PCT` | Emergency stop drawdown | 0.20 |

## Running the Bot

```bash
# Paper trading (default)
python -m src.main --mode paper

# Semi-automatic (requires confirmation)
python -m src.main --mode semi

# Fully automated
python -m src.main --mode auto

# Scan markets without trading
python -m src.main scan

# View status
python -m src.main status

# Backtest
python -m src.main backtest --city nyc --start 2026-01-01 --end 2026-01-15
```

## Code Conventions

### Style
- Python 3.11+ with type hints
- Ruff for linting (`ruff check .`)
- Line length: 100 characters
- Async/await for all I/O operations

### Patterns
- Dataclasses for data structures
- Pydantic for configuration validation
- Structlog for JSON logging
- Context managers for resource cleanup

### Naming
- `snake_case` for functions/variables
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- Descriptive names over abbreviations

## Architecture

### Data Flow
```
Open-Meteo API ─┐
                ├─→ WeatherAggregator ─→ AggregatedForecast
Tomorrow.io ────┤                              │
METAR API ──────┘                              ▼
                                       FairValueCalculator
                                               │
                                               ▼
Polymarket Gamma API ─→ MarketScanner ─→ EdgeDetector
                              │                │
                              ▼                ▼
                        WeatherMarket    TradeSignal
                                               │
                                               ▼
                                        RiskManager
                                               │
                                               ▼
                                      ExecutionEngine
                                               │
                                               ▼
                                   PolymarketClient (CLOB)
```

### Key Components

| Component | Responsibility |
|-----------|---------------|
| `WeatherAggregator` | Combines forecasts from multiple sources |
| `MarketScanner` | Discovers active weather markets |
| `FairValueCalculator` | Computes probability from ensemble |
| `EdgeDetector` | Identifies mispriced markets |
| `ArbitrageDetector` | Finds rebalancing opportunities |
| `RiskManager` | Enforces position/loss limits |
| `ExecutionEngine` | Manages orders and positions |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_strategy.py -v

# Run with logging
pytest -s --log-cli-level=INFO
```

## Key Trading Logic

```python
# Fair value = probability mass in bucket from ensemble forecast
# Example: 30% of ensemble members predict 68-69°F

# Edge calculation
edge = (fair_value - market_price) / market_price

# Trade if: edge > 10% AND model_agreement > 65%

# Position sizing: Half-Kelly with 2% max
kelly = (b * p - q) / b  # where b=odds, p=prob win, q=prob lose
size = min(kelly / 2, 0.02) * bankroll

# Rebalancing arbitrage
# If sum of all bucket YES prices < $1.00, buy all = guaranteed profit
```

## Supported Cities

| Key | City | METAR | Unit |
|-----|------|-------|------|
| nyc | New York City | KNYC | °F |
| london | London | EGLL | °C |
| seoul | Seoul | RKSI | °C |
| dallas | Dallas | KDAL | °F |
| toronto | Toronto | CYYZ | °C |
| seattle | Seattle | KSEA | °F |
| atlanta | Atlanta | KATL | °F |

## Common Tasks for AI Assistants

### Adding a New City
1. Add entry to `CITIES` dict in `src/config.py`
2. Include lat/lon, METAR station code, unit, and timezone
3. Update patterns in `MarketScanner.CITY_PATTERNS`

### Adding a Weather Data Source
1. Create new client in `src/data/`
2. Add to `WeatherAggregator._combine_forecasts()`
3. Adjust source weights as needed

### Modifying Risk Parameters
1. Update defaults in `src/config.py` Settings class
2. Adjust `RiskManager` check methods if needed
3. Update `.env.example` documentation

## Security Considerations

- **Never commit** `.env` files or private keys
- **Paper trade first** before enabling live trading
- Private keys are only needed for `semi` or `auto` modes
- All API keys should be stored in environment variables
- SQLite database may contain position data - handle securely

## Dependencies

| Package | Purpose |
|---------|---------|
| aiohttp | Async HTTP client |
| pydantic | Data validation |
| pydantic-settings | Environment config |
| structlog | JSON logging |
| py-clob-client | Polymarket CLOB API |
| numpy | Numerical operations |
| scipy | Statistical functions |
| aiosqlite | Async SQLite |

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-01-23 | Initial CLAUDE.md creation | Claude AI |
| 2026-01-23 | Full implementation of trading bot | Claude AI |
