# Quick Start Guide

Get the Polymarket Alpha Bot running in 5 minutes.

## Prerequisites

- Python 3.10+
- API Keys (at least one LLM provider)

## 1. Install Dependencies

```bash
cd polymarket-bot
pip install -r requirements.txt
```

## 2. Configure API Keys

```bash
cp config/config.example.yaml config/config.yaml
```

Edit `config/config.yaml` and add your API keys:

### Required (pick one):
```yaml
credentials:
  llm:
    anthropic_api_key: "sk-ant-..."  # Get from console.anthropic.com
    # OR
    openai_api_key: "sk-..."  # Get from platform.openai.com
```

### Recommended:
```yaml
credentials:
  twitter:
    bearer_token: "AAAA..."  # Get from developer.twitter.com
  
  news:
    newsapi_key: "..."  # Get from newsapi.org (free tier works)
```

### For Live Trading (later):
```yaml
credentials:
  polymarket:
    private_key: "0x..."  # Your wallet private key (BE CAREFUL!)
```

## 3. Test the System

### Test signal processing:
```bash
python cli.py test-signal "BREAKING: Fed announces 50bp rate cut" --analyze
```

### Fetch live markets:
```bash
python cli.py fetch-markets --show 10
```

### Search for specific markets:
```bash
python cli.py search-markets "bitcoin"
```

### Run a backtest:
```bash
python cli.py backtest --days 30 --capital 10000
```

## 4. Run in Paper Trading Mode

```bash
python main.py --paper
```

This will:
- Monitor configured signal sources
- Analyze signals with LLM
- Make paper trades (simulated)
- Log everything to `logs/bot.log`

## 5. Monitor the Bot

### Start monitoring server:
```bash
python cli.py monitor --port 8000
```

Then open: http://localhost:8000

### Or run bot with monitoring:
```bash
python main.py --paper  # Port 8000 by default
```

## 6. Understanding the Output

### Signal Processing:
```
[INFO] signal_received source=twitter content="BREAKING: ..."
[INFO] signal_matched market_id=abc123 confidence=0.85
[INFO] trade_opportunity edge_percent=25.3
[INFO] trade_executed side=yes amount=100 paper_trade=True
```

### Key Metrics:
- **Edge**: Difference between estimated probability and current price
- **Confidence**: How sure the LLM is about the match
- **Priority Score**: How important the signal is (0-100)

## 7. Going Live (When Ready)

⚠️ **Only after extensive paper trading!**

1. Ensure your wallet has USDC on Polygon
2. Update config:
```yaml
trading:
  enabled: true
  paper_trading: false
  max_position_size_usd: 100  # Start small!
```

3. Run:
```bash
python main.py
```

## Common Issues

### "No markets fetched"
- Polymarket API might be down
- Check your internet connection

### "LLM analysis failed"
- Check API key is valid
- Check you have credits/quota

### "Signal not actionable"
- Signal didn't match any markets
- Try signals with clear market implications

### "Trade rejected"
- Edge below threshold (default 15%)
- Position limit reached
- Market cooldown active

## File Structure

```
polymarket-bot/
├── config/
│   └── config.yaml      # Your configuration
├── data/
│   └── bot.db           # SQLite database
├── logs/
│   └── bot.log          # Log files
├── src/
│   ├── ingestion/       # Signal monitors
│   ├── processing/      # LLM & prioritization
│   ├── markets/         # Polymarket client
│   ├── execution/       # Trading logic
│   └── utils/           # Helpers
├── main.py              # Main entry point
└── cli.py               # CLI tools
```

## Next Steps

1. **Customize signal sources** in config - add accounts/keywords relevant to markets you're interested in

2. **Tune thresholds** - adjust `min_edge_percent` and `kelly_fraction` based on your risk tolerance

3. **Add more monitors** - Telegram, custom RSS feeds, on-chain tracking

4. **Review logs** - Understand why signals are/aren't triggering trades

5. **Backtest strategies** - Use the backtesting module to test different approaches

## Getting Help

- Check logs: `tail -f logs/bot.log`
- View metrics: http://localhost:8000/metrics
- API health: http://localhost:8000/health
