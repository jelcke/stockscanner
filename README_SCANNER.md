# IB Stock Scanner - User Guide

## Overview

The IB Stock Scanner is a real-time stock scanning system that connects to Interactive Brokers (IB) to monitor stocks based on configurable criteria including price movements, volume surges, technical indicators, and more.

## Prerequisites

1. **Interactive Brokers Account** with API access enabled
2. **IB Gateway or TWS** running with API connections enabled
3. **Python 3.9+** installed
4. **Redis** (optional, for enhanced caching)

## Installation

### 1. Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings:
# - IB_HOST=127.0.0.1
# - IB_PORT=7497 (paper trading) or 7496 (live trading)
# - REDIS_URL=redis://localhost:6379 (optional)
# - TELEGRAM_BOT_TOKEN=... (for alerts)
```

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Note: TA-Lib requires system dependencies
# On Ubuntu/Debian:
sudo apt-get install ta-lib
# On macOS:
brew install ta-lib
```

### 3. Configure IB Gateway/TWS

1. Open IB Gateway or TWS
2. Go to Configure → Settings → API → Settings
3. Enable "Enable ActiveX and Socket Clients"
4. Set Socket port (7497 for paper, 7496 for live)
5. Add 127.0.0.1 to "Trusted IP Addresses"
6. Disable "Read-Only API"

## Basic Commands

### Test Connection
```bash
# Test connection to IB Gateway
uv run python -m src.main test-connection

# With custom settings
uv run python -m src.main test-connection --host 127.0.0.1 --port 7497
```

### Run a Scan
```bash
# Scan with default momentum criteria
uv run python -m src.main scan

# Scan specific symbols
uv run python -m src.main scan --symbols AAPL,MSFT,GOOGL

# Use different criteria
uv run python -m src.main scan --criteria volume     # Volume breakouts
uv run python -m src.main scan --criteria oversold   # Oversold bounce
uv run python -m src.main scan --criteria technical  # Technical breakout
```

### Continuous Scanning
```bash
# Run continuous scan (updates every 60 seconds)
uv run python -m src.main scan --continuous

# Custom interval (30 seconds)
uv run python -m src.main scan --continuous --interval 30

# Scan for specific duration (1 hour)
uv run python -m src.main scan --duration 3600
```

### Export Results
```bash
# Export scan results to CSV
uv run python -m src.main scan --export results.csv
```

## Configuration

### Edit config.yaml

Key sections to configure:

```yaml
# Price criteria
criteria:
  - type: "price"
    min_price: 1.0      # Minimum stock price
    max_price: 20.0     # Maximum stock price
    min_change_pct: 1.0 # Minimum % change

# Volume criteria
  - type: "volume"
    min_volume: 100000
    volume_surge_multiple: 2.0  # 2x average volume

# Alerts
alerts:
  conditions:
    volume_surge: 3.0    # Alert when volume > 3x average
    price_change: 5.0    # Alert on 5% price move
```

### Universes

Available stock universes:
- `US_STOCKS` - All US stocks
- `TECH` - Technology stocks
- `SP500` - S&P 500 components
- `TEST` - Test set (AAPL, MSFT, GOOGL)

```bash
uv run python -m src.main scan --universe TECH
```

## Scanning Criteria

### Momentum Gainers
- RSI between 50-70
- Price above 20-day SMA
- Positive price change

### Volume Breakouts
- Volume 3x above average
- Price change > 2%
- High volume ratio

### Oversold Bounce
- RSI < 35
- Price near support
- Volume increasing

### Technical Breakout
- Price above upper Bollinger Band
- MACD bullish crossover
- Strong momentum

## Advanced Features

### Backtesting
```bash
# Backtest criteria on historical data
uv run python -m src.main backtest --symbols AAPL,MSFT --criteria momentum --days 30
```

### Database Management
```bash
# View database statistics
uv run python -m src.main stats

# Clean up old data (older than 30 days)
uv run python -m src.main cleanup
```

### Alert Configuration

#### Telegram Alerts
1. Create bot with @BotFather on Telegram
2. Get bot token and chat ID
3. Add to .env:
   ```
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

#### Email Alerts
Configure in config.yaml:
```yaml
alerts:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    from_address: "scanner@gmail.com"
    to_addresses: ["trader@gmail.com"]
```

## Performance Tips

1. **Rate Limiting**: The scanner respects IB's 50 messages/second limit
2. **Market Data Lines**: Limited to 100 concurrent subscriptions (standard account)
3. **Caching**: Use Redis for better performance with large universes
4. **Batch Size**: Adjust batch processing in config.yaml

## Troubleshooting

### Connection Issues
- Ensure IB Gateway/TWS is running
- Check API settings are enabled
- Verify host/port in .env match IB Gateway settings
- Check firewall isn't blocking connections

### No Results
- Check market hours (scanner works during market hours)
- Verify criteria aren't too restrictive
- Use `--log-level DEBUG` for detailed logs

### Common Errors
- **"No market data permissions"**: Need market data subscription for real-time data
- **"Max market data lines exceeded"**: Reduce universe size or upgrade IB account
- **"Connection refused"**: IB Gateway not running or wrong port

### Debug Mode
```bash
# Run with debug logging
uv run python -m src.main --log-level DEBUG scan

# Check specific symbol
uv run python -m src.main scan --symbols AAPL --log-level DEBUG
```

## Example Workflow

1. **Morning Setup**
   ```bash
   # Test connection
   uv run python -m src.main test-connection
   
   # Run initial scan
   uv run python -m src.main scan --universe TECH --criteria momentum
   ```

2. **Continuous Monitoring**
   ```bash
   # Start continuous scanner
   uv run python -m src.main scan --continuous --interval 30 --criteria volume
   ```

3. **Export Results**
   ```bash
   # Export interesting stocks
   uv run python -m src.main scan --export daily_picks.csv
   ```

4. **End of Day**
   ```bash
   # View statistics
   uv run python -m src.main stats
   
   # Clean old data
   uv run python -m src.main cleanup
   ```

## Next Steps

- **Enable ML Features**: Train models for pattern detection (coming soon)
- **Add Sentiment Analysis**: Configure news/social APIs in .env
- **Custom Indicators**: Extend scanner with custom technical indicators
- **Production Deployment**: Set up as systemd service for 24/7 operation

For more details, see the individual component documentation in the `docs/` directory.