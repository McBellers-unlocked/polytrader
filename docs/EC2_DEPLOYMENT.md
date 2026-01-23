# Deploying Polytrader to AWS EC2

This guide walks you through deploying Polytrader on an AWS EC2 instance with a web dashboard.

## Prerequisites

- AWS Account
- Basic familiarity with AWS Console

---

## Step 1: Launch an EC2 Instance

### 1.1 Go to EC2 Dashboard
1. Log into [AWS Console](https://console.aws.amazon.com)
2. Search for "EC2" in the search bar
3. Click "Launch Instance"

### 1.2 Configure the Instance

| Setting | Value |
|---------|-------|
| **Name** | `polytrader` |
| **AMI** | Ubuntu Server 24.04 LTS (free tier eligible) |
| **Instance type** | `t2.micro` (free tier) or `t3.small` for better performance |
| **Key pair** | Create new or select existing (you'll need this to SSH in) |

### 1.3 Network Settings
Click "Edit" on Network settings and add these rules:

| Type | Port | Source | Description |
|------|------|--------|-------------|
| SSH | 22 | My IP | SSH access |
| Custom TCP | 8000 | 0.0.0.0/0 | Web dashboard |

### 1.4 Storage
- 20 GB is enough (default 8 GB works too)

### 1.5 Launch
Click "Launch Instance" and wait for it to start.

---

## Step 2: Connect to Your Instance

### 2.1 Find Your Instance IP
1. Go to EC2 > Instances
2. Click on your instance
3. Copy the "Public IPv4 address" (e.g., `54.123.45.67`)

### 2.2 Connect via SSH

**On Mac/Linux:**
```bash
# Make your key file secure
chmod 400 your-key.pem

# Connect
ssh -i your-key.pem ubuntu@YOUR_IP_ADDRESS
```

**On Windows:**
- Use PuTTY or Windows Terminal with the same command

---

## Step 3: Install Dependencies

Run these commands on your EC2 instance:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Verify Python version
python3.11 --version
```

---

## Step 4: Deploy Polytrader

```bash
# Clone the repository (replace with your repo URL)
git clone https://github.com/YOUR_USERNAME/polytrader.git
cd polytrader

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
pip install rich uvicorn fastapi jinja2

# Verify installation
python -m src.main --help
```

---

## Step 5: Configure Environment

```bash
# Create configuration file
cat > .env << 'EOF'
# Trading Mode
TRADING_MODE=paper

# Bankroll
STARTING_BANKROLL=1500.0

# Risk Parameters
MAX_POSITION_PCT=0.02
DAILY_LOSS_STOP_PCT=0.05
MAX_DRAWDOWN_PCT=0.20

# Thresholds
MIN_EDGE_THRESHOLD=0.10
MIN_MODEL_AGREEMENT=0.65

# Logging
LOG_LEVEL=INFO

# For live trading (uncomment and fill in):
# POLYMARKET_PRIVATE_KEY=your_key_here
# POLYMARKET_FUNDER=your_funder_address
# POLYMARKET_API_KEY=your_api_key
EOF
```

---

## Step 6: Test the Bot

```bash
# Activate virtual environment (if not already)
source venv/bin/activate

# Run with mock data
python -m src.main --mode paper --mock

# Press Ctrl+C to stop after a few iterations
```

You should see opportunities being detected and trades being logged.

---

## Step 7: Start the Web Dashboard

```bash
# Start web server (runs on port 8000)
python -m src.main web
```

Now open your browser and go to:
```
http://YOUR_EC2_IP:8000
```

You should see the Polytrader dashboard!

---

## Step 8: Run as Background Service (Production)

### 8.1 Create a systemd service file

```bash
sudo nano /etc/systemd/system/polytrader.service
```

Paste this content:
```ini
[Unit]
Description=Polytrader Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/polytrader
Environment=PATH=/home/ubuntu/polytrader/venv/bin
ExecStart=/home/ubuntu/polytrader/venv/bin/python -m src.main web --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save with `Ctrl+X`, then `Y`, then `Enter`.

### 8.2 Enable and start the service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable polytrader

# Start the service
sudo systemctl start polytrader

# Check status
sudo systemctl status polytrader
```

### 8.3 View logs

```bash
# View recent logs
sudo journalctl -u polytrader -f
```

---

## Step 9: Run Bot Separately (Optional)

If you want to run the trading bot separately from the web dashboard:

### Terminal 1: Web Dashboard
```bash
python -m src.main web
```

### Terminal 2: Trading Bot
```bash
python -m src.main --mode paper --mock
```

Or create two systemd services (one for web, one for bot).

---

## Security Tips

1. **Never commit `.env` files** to git
2. **Use paper mode first** until you're confident
3. **Set up AWS billing alerts** to avoid surprise charges
4. **Consider using a separate AWS account** for trading

---

## Useful Commands

```bash
# Check if service is running
sudo systemctl status polytrader

# Restart service
sudo systemctl restart polytrader

# Stop service
sudo systemctl stop polytrader

# View logs
sudo journalctl -u polytrader -f

# Check database
sqlite3 polytrader.db "SELECT COUNT(*) FROM opportunities;"
```

---

## Troubleshooting

### Can't connect to port 8000?
- Check EC2 Security Group has port 8000 open
- Make sure the service is running: `sudo systemctl status polytrader`

### Service won't start?
- Check logs: `sudo journalctl -u polytrader -n 50`
- Verify paths in the service file

### Database locked?
- Only one process can write at a time
- Stop the bot before running manual queries

---

## Cost Estimate

| Resource | Monthly Cost |
|----------|--------------|
| t2.micro | Free (750 hrs/month for 12 months) |
| t3.small | ~$15/month |
| Storage (20GB) | ~$2/month |

**Free tier eligible for first year!**

---

## Next Steps

1. Monitor the dashboard for a few days in paper mode
2. Analyze the trading patterns
3. Adjust risk parameters if needed
4. When ready, switch to `semi` mode for manual confirmation
5. Eventually move to `auto` mode for full automation

Good luck!
