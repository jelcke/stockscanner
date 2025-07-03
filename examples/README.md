# Interactive Brokers Stock Scanner Examples

This directory contains comprehensive examples demonstrating various features of an AI-enhanced stock scanner using the Interactive Brokers API.

## Prerequisites

1. **Interactive Brokers Account** (paper trading is fine)
2. **IB Gateway or TWS** installed and running
3. **API Access Enabled** in TWS/Gateway settings
4. **Python 3.8+** with dependencies installed

## Setup

1. Install dependencies:
```bash
uv venv .venv
uv pip install -r requirements.txt
```

2. Configure IB Gateway/TWS:
   - Enable API access in Configuration
   - Set socket port (default: 7497 for paper trading)
   - Allow connections from localhost

3. Update configuration in `config/scanner_config.yaml` if needed

## Examples Overview

### 1. Basic Scanner (`basic_scanner.py`)
Demonstrates core IB API connectivity and simple percentage gain filtering.

```bash
uv run python basic_scanner.py
```

Features:
- Connect to IB Gateway
- Subscribe to real-time/delayed market data
- Track price changes and volume
- Display top gainers

### 2. Pattern Detection (`pattern_detection.py`)
Uses computer vision and technical analysis to detect chart patterns.

Features:
- Cup and Handle detection
- Ascending Triangle patterns
- Bull Flag identification
- Double Bottom recognition

### 3. Sentiment Integration (`sentiment_integration.py`)
Combines news sentiment analysis with technical scanning.

Features:
- News API integration
- Sentiment scoring
- Momentum analysis
- Alert generation

### 4. ML Breakout Predictor (`ml_breakout_predictor.py`)
Machine learning model to predict potential breakout stocks.

Features:
- Feature engineering (RSI, MACD, Bollinger Bands, etc.)
- Random Forest/Gradient Boosting models
- Breakout probability calculation
- Model persistence

### 5. Advanced Display (`advanced_display.py`)
Professional terminal interface with rich formatting.

Features:
- Live updating tables
- Color-coded alerts
- Multi-panel layout
- Real-time statistics

### 6. Backtesting Framework (`backtesting_framework.py`)
Historical validation system for scanner strategies.

Features:
- Strategy backtesting
- Performance metrics
- Risk analysis
- Visual reports

## Demo Mode

If you don't have IB Gateway running, use the demo scanner:

```bash
uv run python demo_scanner.py
```

This simulates market data without requiring an IB connection.

## Common Issues

### Connection Refused
- Ensure IB Gateway/TWS is running
- Check port settings (7497 for paper, 7496 for live)
- Verify API access is enabled

### No Market Data
- Paper accounts use delayed data (15-20 min)
- Check market hours
- Verify symbol validity

### Missing Dependencies
- Run `uv pip install -r requirements.txt`
- Some examples require additional setup (API keys, etc.)

## Configuration

Edit `config/scanner_config.yaml` to customize:
- IB connection settings
- Scanner criteria
- Alert thresholds
- API keys for external services

## Testing Connection

Use the test script to verify your setup:

```bash
uv run python test_ib_connection.py
```

## Notes

- Examples use asyncio for efficient concurrent operations
- Delayed market data is free but has 15-20 minute delay
- Real-time data requires market data subscriptions
- Be mindful of API rate limits

## Further Development

These examples provide a foundation for building a comprehensive trading system. Consider:
- Adding database persistence
- Implementing trading logic
- Creating web interfaces
- Adding more ML models
- Integrating additional data sources

For questions or issues, check the Interactive Brokers API documentation or the ib_async library documentation.