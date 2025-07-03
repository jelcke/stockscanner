name: "Interactive Brokers Real-Time Stock Scanner with AI-Enhanced Pattern Recognition"
description: |

## Purpose
Build a production-ready Python stock scanner that connects to Interactive Brokers API to identify fast-moving stocks with high percentage gains, unusual volume activity, and momentum breakouts. The system will include AI-powered pattern recognition, sentiment analysis integration, and machine learning models for predicting breakout probability.

## Core Principles
1. **Production Quality**: Handle real-time data streams, scale to hundreds of symbols, proper error recovery
2. **Modular Architecture**: Clear separation of concerns, reusable components, testable units
3. **AI/ML Integration**: Pattern recognition, breakout prediction, sentiment analysis
4. **Performance**: Async operations, efficient data processing, proper caching
5. **Global rules**: Follow all rules in CLAUDE.md, use uv for Python, venv_linux environment

---

## Goal
Create a fully functional stock scanner that:
- Connects to IB API for real-time market data
- Scans stocks based on configurable criteria (price movement, volume, technical indicators)
- Uses AI/ML for pattern recognition and breakout prediction
- Integrates sentiment analysis from news and social media
- Provides real-time alerts via multiple channels
- Displays results in a rich terminal interface
- Includes comprehensive testing and error handling

## Why
- **Traders need real-time insights**: Identify opportunities as they happen
- **AI enhancement**: Go beyond simple technical indicators to pattern recognition
- **Multi-signal validation**: Combine price action, volume, sentiment for better signals
- **Automation**: Save hours of manual screening
- **Backtesting**: Validate strategies before live trading

## What
### User-visible behavior
- Terminal UI showing real-time scanning results
- Configurable scan criteria via YAML
- Multiple alert channels (email, Telegram, webhooks)
- Historical backtesting capabilities
- Export results to CSV/JSON

### Technical requirements
- Handle IB API rate limits (50 msg/sec)
- Manage market data lines efficiently (100 standard limit)
- Process streaming data without memory leaks
- Thread-safe callback handling
- Secure credential management
- < 100ms inference time for ML models

### Success Criteria
- [ ] Connects to IB Gateway/TWS and maintains stable connection
- [ ] Scans 200+ symbols concurrently without hitting rate limits
- [ ] ML pattern detection works on real-time data
- [ ] Sentiment analysis integrates with at least 2 sources
- [ ] Alerts fire within 5 seconds of criteria match
- [ ] All tests pass with >80% coverage
- [ ] Handles disconnections/errors gracefully

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://ib-async.readthedocs.io/api.html
  why: Complete ib_async API reference - async IB connection patterns
  critical: Study IB.connectAsync(), reqMktData(), qualifyContractsAsync()
  
- url: https://interactivebrokers.github.io/tws-api/market_data.html
  why: IB market data subscriptions and limitations
  critical: Understand delayed data type 3, market data lines, rate limits

- url: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  why: ML model for breakout prediction
  critical: Feature importance, hyperparameter tuning
  
- url: https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
  why: Chart pattern recognition from candlestick images
  critical: Converting price data to images for CNN

- file: examples/basic_scanner.py
  why: Core pattern for IB connection, market data subscription, event handling
  critical: Copy async patterns, error handling, symbol qualification
  
- file: examples/ml_breakout_predictor.py
  why: ML integration pattern, feature engineering approach
  critical: Model loading, inference pipeline, confidence scoring

- file: examples/utils/data_manager.py
  why: Data validation, caching patterns, SQLite integration
  critical: DataCache implementation, OHLCV validation

- file: examples/config/scanner_config.yaml
  why: Configuration structure and defaults
  critical: All configurable parameters and their types

- doc: https://newsapi.org/docs/endpoints/everything
  section: Request parameters
  critical: API key in headers, 100 requests/day free tier

- doc: https://github.com/pushshift/api
  section: Reddit data access
  critical: Rate limits, data structure for sentiment analysis
```

### Current Codebase tree
```bash
stockscanner/
├── CLAUDE.md
├── INITIAL.md
├── PLANNING.md
├── README.md
├── TASK.md
├── PRPs/
│   ├── ib-stock-scanner.md
│   └── templates/
│       └── prp_base.md
└── examples/
    ├── README.md
    ├── advanced_display.py
    ├── backtesting_framework.py
    ├── basic_scanner.py
    ├── config/
    │   ├── model_config.yaml
    │   └── scanner_config.yaml
    ├── demo_scanner.py
    ├── ml_breakout_predictor.py
    ├── pattern_detection.py
    ├── requirements.txt
    ├── run_examples.py
    ├── sentiment_integration.py
    ├── test_ib_connection.py
    └── utils/
        ├── __init__.py
        ├── data_manager.py
        ├── indicators.py
        └── patterns.py
```

### Desired Codebase tree with files to be added
```bash
stockscanner/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Entry point with CLI
│   ├── scanner/
│   │   ├── __init__.py
│   │   ├── scanner.py             # Main scanner orchestration
│   │   ├── ib_connection.py      # IB API connection management
│   │   ├── data_processor.py     # Real-time data processing
│   │   └── criteria.py           # Scanning criteria definitions
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── pattern_detector.py   # Chart pattern recognition
│   │   ├── breakout_predictor.py # ML breakout prediction
│   │   ├── feature_engineering.py # Feature extraction
│   │   └── model_manager.py      # Model loading/inference
│   ├── sentiment/
│   │   ├── __init__.py
│   │   ├── news_analyzer.py      # News sentiment analysis
│   │   ├── social_analyzer.py    # Reddit/Twitter sentiment
│   │   └── sentiment_aggregator.py # Combined sentiment
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cache_manager.py      # Redis caching layer
│   │   ├── database.py           # SQLite operations
│   │   ├── models.py             # SQLAlchemy models
│   │   └── validators.py         # Data validation
│   ├── alerts/
│   │   ├── __init__.py
│   │   ├── alert_manager.py      # Alert orchestration
│   │   ├── channels.py           # Email/Telegram/Webhook
│   │   └── display.py            # Rich terminal UI
│   └── config/
│       ├── __init__.py
│       ├── settings.py           # Config management
│       └── constants.py          # System constants
├── tests/
│   ├── __init__.py
│   ├── test_scanner.py
│   ├── test_ml.py
│   ├── test_sentiment.py
│   ├── test_data.py
│   ├── test_alerts.py
│   └── fixtures/
│       └── mock_data.py
├── models/                       # Trained ML models
├── data/                        # SQLite DB, cache files
├── logs/                        # Application logs
├── exports/                     # CSV exports
├── config.yaml                  # User configuration
├── requirements.txt
├── .env.example
└── pyproject.toml
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: IB API Rate Limiting
# ib_async does NOT automatically handle rate limits
# Must implement queue with max 50 messages/second
import asyncio
from collections import deque
rate_limiter = deque(maxlen=50)  # Track last 50 request timestamps

# CRITICAL: Market Data Lines
# Each symbol uses 1 line, standard accounts get 100
# Must track active subscriptions and unsubscribe when not needed
active_subscriptions = {}  # {symbol: ticker}

# CRITICAL: Symbol Qualification
# Not all symbols are valid contracts, MUST qualify first
contracts = await ib.qualifyContractsAsync(Stock('AAPL', 'SMART', 'USD'))
if not contracts:
    # Handle invalid symbol

# CRITICAL: Connection Management
# IB Gateway disconnects on inactivity or errors
# Implement exponential backoff reconnection
async def connect_with_retry(max_attempts=5):
    for attempt in range(max_attempts):
        try:
            await ib.connectAsync(host, port, clientId)
            return
        except:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

# CRITICAL: Thread Safety
# IB callbacks run in separate threads
# Use asyncio.run_coroutine_threadsafe() for thread-safe updates
def on_ticker_update(ticker):
    asyncio.run_coroutine_threadsafe(
        process_ticker(ticker), 
        loop
    )

# CRITICAL: Memory Management
# Continuous data streams cause memory leaks
# Use circular buffers for price history
from collections import deque
price_history = deque(maxlen=1000)  # Keep last 1000 ticks

# CRITICAL: pandas with asyncio
# pandas operations are blocking, wrap in executor
df = await loop.run_in_executor(None, pd.read_csv, 'data.csv')

# CRITICAL: ML Model Thread Safety
# TensorFlow/Keras models aren't thread-safe
# Load once, use locks for inference
import threading
model_lock = threading.Lock()

# CRITICAL: Redis Connection Pooling
# Don't create new connections per operation
import redis
redis_pool = redis.ConnectionPool(host='localhost', port=6379, max_connections=50)
redis_client = redis.Redis(connection_pool=redis_pool)

# CRITICAL: Time Zones
# IB returns data in exchange timezone
# Always convert to UTC for storage
from datetime import timezone
utc_time = exchange_time.replace(tzinfo=exchange_tz).astimezone(timezone.utc)
```

## Implementation Blueprint

### Data models and structure
```python
# src/data/models.py - SQLAlchemy models
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Stock(Base):
    __tablename__ = 'stocks'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, index=True)
    company_name = Column(String(100))
    sector = Column(String(50))
    exchange = Column(String(20))
    
class PriceData(Base):
    __tablename__ = 'price_data'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    
class ScanResult(Base):
    __tablename__ = 'scan_results'
    id = Column(Integer, primary_key=True)
    scan_time = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(10))
    criteria_matched = Column(String(200))
    price = Column(Float)
    volume = Column(Integer)
    change_pct = Column(Float)
    breakout_probability = Column(Float)
    sentiment_score = Column(Float)

# src/scanner/criteria.py - Pydantic models
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum

class CriteriaType(str, Enum):
    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    TECHNICAL = "technical"
    
class PriceCriteria(BaseModel):
    min_price: Optional[float] = Field(None, ge=0)
    max_price: Optional[float] = Field(None, ge=0)
    min_change_pct: Optional[float] = None
    max_change_pct: Optional[float] = None
    
    @validator('max_price')
    def validate_price_range(cls, v, values):
        if v and values.get('min_price') and v < values['min_price']:
            raise ValueError('max_price must be >= min_price')
        return v

class VolumeCriteria(BaseModel):
    min_volume: Optional[int] = Field(None, ge=0)
    min_avg_volume: Optional[int] = Field(None, ge=0)
    volume_surge_multiple: Optional[float] = Field(None, ge=1.0)

class ScannerConfig(BaseModel):
    criteria: List[Union[PriceCriteria, VolumeCriteria]]
    max_results: int = Field(100, ge=1, le=1000)
    scan_interval: int = Field(60, ge=5)
```

### List of tasks to complete the PRP

```yaml
Task 1: Create project structure and configuration
CREATE src/__init__.py:
  - Empty file for package initialization
CREATE src/config/constants.py:
  - PATTERN from: examples/basic_scanner.py lines 27-32
  - Define IB connection defaults, rate limits, market data constants
CREATE src/config/settings.py:
  - Use python-dotenv to load .env
  - Load and validate config.yaml using pydantic
  - PATTERN: hierarchical config like examples/config/scanner_config.yaml

Task 2: Implement data models and database
CREATE src/data/models.py:
  - SQLAlchemy models as shown in blueprint above
  - Add __repr__ methods for debugging
CREATE src/data/database.py:
  - PATTERN from: examples/utils/data_manager.py lines 149-200
  - Connection management, session handling
  - CRUD operations for stocks, price data, scan results
CREATE src/data/validators.py:
  - PATTERN from: examples/utils/data_manager.py DataValidator class
  - Validate OHLCV data, check for outliers
  - Ensure data integrity before storage

Task 3: Build IB connection manager
CREATE src/scanner/ib_connection.py:
  - PATTERN from: examples/basic_scanner.py connect() method
  - Implement connection with exponential backoff retry
  - Handle disconnections with auto-reconnect
  - Track market data lines usage
  - Rate limiting queue implementation

Task 4: Create core scanner
CREATE src/scanner/scanner.py:
  - PATTERN from: examples/basic_scanner.py BasicScanner class
  - Orchestrate scanning process
  - Symbol universe management
  - Concurrent symbol processing with asyncio.gather
  - Result aggregation and ranking
CREATE src/scanner/data_processor.py:
  - Real-time ticker processing
  - Calculate technical indicators
  - Volume analysis (relative volume, surges)
  - Price movement detection

Task 5: Implement scanning criteria
CREATE src/scanner/criteria.py:
  - Pydantic models as shown in blueprint
  - Criteria evaluation engine
  - Composite criteria support (AND/OR)
  - Preset criteria loading from config

Task 6: Add ML pattern detection
CREATE src/ml/feature_engineering.py:
  - PATTERN from: examples/ml_breakout_predictor.py lines 70-80
  - Technical indicator calculation
  - Price/volume features
  - Rolling statistics
CREATE src/ml/pattern_detector.py:
  - PATTERN from: examples/pattern_detection.py
  - Convert OHLCV to candlestick images
  - CNN model for pattern classification
  - Pattern types: cup-handle, triangles, flags
CREATE src/ml/breakout_predictor.py:
  - Random Forest classifier
  - Feature importance analysis
  - Probability calibration
  - Model versioning

Task 7: Integrate sentiment analysis
CREATE src/sentiment/news_analyzer.py:
  - NewsAPI integration
  - Title and description parsing
  - Sentiment scoring with TextBlob
  - Handle API rate limits
CREATE src/sentiment/social_analyzer.py:
  - Reddit API via PRAW
  - Subreddit monitoring (wallstreetbets, stocks)
  - Comment sentiment extraction
CREATE src/sentiment/sentiment_aggregator.py:
  - Weighted sentiment combination
  - Time-decay for older sentiment
  - Sentiment momentum calculation

Task 8: Build alert system
CREATE src/alerts/alert_manager.py:
  - Alert rule engine
  - Deduplication (don't spam same alert)
  - Alert history tracking
  - Priority levels
CREATE src/alerts/channels.py:
  - Email alerts via SMTP
  - Telegram bot integration
  - Webhook dispatcher
  - Channel-specific formatting

Task 9: Create display interface
CREATE src/alerts/display.py:
  - PATTERN from: examples/advanced_display.py
  - Rich tables for scan results
  - Color coding (green gains, red losses)
  - Real-time updates without flicker
  - Progress bars for scan status

Task 10: Implement caching layer
CREATE src/data/cache_manager.py:
  - PATTERN from: examples/utils/data_manager.py DataCache class
  - Redis integration for real-time data
  - TTL-based expiration
  - Cache warming strategies

Task 11: Add main entry point
CREATE src/main.py:
  - Click CLI interface
  - Command: scan, backtest, train-model
  - Graceful shutdown handling
  - Error reporting

Task 12: Create comprehensive tests
CREATE tests/test_scanner.py:
  - Mock IB API responses
  - Test criteria evaluation
  - Connection retry logic
CREATE tests/test_ml.py:
  - Feature engineering validation
  - Model prediction tests
  - Pattern detection accuracy
CREATE tests/fixtures/mock_data.py:
  - Sample OHLCV data
  - Mock ticker events
  - Test scan results

Task 13: Add configuration files
CREATE config.yaml:
  - PATTERN from: examples/config/scanner_config.yaml
  - User-editable settings
  - Default scan criteria
CREATE .env.example:
  - IB_HOST=127.0.0.1
  - IB_PORT=7497
  - NEWS_API_KEY=your_key_here
  - REDIS_URL=redis://localhost:6379
CREATE requirements.txt:
  - Copy from examples/requirements.txt
  - Add redis, sqlalchemy, pydantic v2
CREATE pyproject.toml:
  - Project metadata
  - Tool configurations (black, ruff, mypy)
```

### Per task pseudocode

```python
# Task 3: IB Connection Manager
# src/scanner/ib_connection.py
class IBConnectionManager:
    def __init__(self, config: dict):
        self.ib = IB()
        self.config = config
        self.rate_limiter = deque(maxlen=50)
        self.active_subscriptions = {}
        self.reconnect_task = None
        
    async def connect(self):
        # PATTERN: Exponential backoff from gotchas
        for attempt in range(self.config['max_reconnect_attempts']):
            try:
                await self.ib.connectAsync(
                    self.config['host'],
                    self.config['port'],
                    clientId=self.config['client_id']
                )
                self.ib.reqMarketDataType(3)  # Delayed data
                # Set up disconnect handler
                self.ib.disconnectedEvent += self._on_disconnect
                return True
            except Exception as e:
                wait_time = min(2 ** attempt, 60)  # Max 60 seconds
                await asyncio.sleep(wait_time)
        return False
    
    async def subscribe_market_data(self, symbol: str):
        # CRITICAL: Check market data lines
        if len(self.active_subscriptions) >= self.config['max_market_data_lines']:
            # Unsubscribe least recently used
            lru_symbol = min(self.active_subscriptions, key=lambda x: x.last_update)
            await self.unsubscribe_market_data(lru_symbol)
            
        # CRITICAL: Rate limiting
        await self._rate_limit()
        
        # CRITICAL: Qualify contract first
        contract = Stock(symbol, 'SMART', 'USD')
        contracts = await self.ib.qualifyContractsAsync(contract)
        if not contracts:
            raise ValueError(f"Cannot qualify {symbol}")
            
        ticker = self.ib.reqMktData(contracts[0])
        self.active_subscriptions[symbol] = ticker
        return ticker

# Task 6: ML Pattern Detector
# src/ml/pattern_detector.py
class ChartPatternDetector:
    def __init__(self, model_path: str):
        # CRITICAL: Thread-safe model loading
        self.model_lock = threading.Lock()
        self.model = self._load_model(model_path)
        
    def detect_patterns(self, ohlcv_data: pd.DataFrame) -> List[PatternResult]:
        # PATTERN: Convert to image from examples/pattern_detection.py
        image = self._create_candlestick_image(ohlcv_data)
        
        # CRITICAL: Thread-safe inference
        with self.model_lock:
            # Preprocess image
            processed = self._preprocess_image(image)
            
            # Run inference
            predictions = self.model.predict(processed)
            
        # Parse predictions
        patterns = []
        for pattern_type, confidence in predictions:
            if confidence > self.confidence_threshold:
                patterns.append(PatternResult(
                    type=pattern_type,
                    confidence=confidence,
                    start_idx=self._find_pattern_start(pattern_type, ohlcv_data),
                    end_idx=len(ohlcv_data) - 1
                ))
        return patterns

# Task 8: Alert Manager
# src/alerts/alert_manager.py
class AlertManager:
    def __init__(self, config: dict):
        self.config = config
        self.channels = self._init_channels()
        self.alert_history = deque(maxlen=1000)
        self.dedup_window = timedelta(minutes=config['dedup_minutes'])
        
    async def check_and_send_alerts(self, scan_results: List[ScanResult]):
        for result in scan_results:
            # Check if should alert
            if not self._should_alert(result):
                continue
                
            # Check deduplication
            if self._is_duplicate(result):
                continue
                
            # Format alert
            alert = self._format_alert(result)
            
            # Send to all enabled channels
            tasks = []
            for channel in self.channels:
                if channel.enabled:
                    tasks.append(channel.send(alert))
                    
            # Send concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Record in history
            self.alert_history.append({
                'symbol': result.symbol,
                'time': datetime.now(),
                'criteria': result.criteria_matched
            })
```

### Integration Points
```yaml
DATABASE:
  - migration: "CREATE TABLE stocks, price_data, scan_results with proper indexes"
  - index: "CREATE INDEX idx_symbol_timestamp ON price_data(symbol, timestamp)"
  - index: "CREATE INDEX idx_scan_time ON scan_results(scan_time)"
  
CONFIG:
  - add to: src/config/settings.py
  - pattern: "IB_HOST = os.getenv('IB_HOST', '127.0.0.1')"
  - pattern: "load config.yaml and merge with env vars"
  
CACHE:
  - add to: src/data/cache_manager.py
  - pattern: "redis_client = redis.Redis.from_url(REDIS_URL)"
  - keys: "scanner:ticker:{symbol}", "scanner:sentiment:{symbol}"
  
MODELS:
  - directory: models/
  - files: "pattern_detector.h5", "breakout_predictor.pkl"
  - versioning: "model_v{timestamp}.pkl"
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
cd /home/jelcke/dev/test/stockscanner
uv run ruff check src/ --fix     # Auto-fix style issues
uv run mypy src/                  # Type checking
uv run black src/                 # Format code

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# tests/test_scanner.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.scanner.ib_connection import IBConnectionManager

@pytest.mark.asyncio
async def test_connection_retry():
    """Test exponential backoff on connection failure"""
    config = {'max_reconnect_attempts': 3, 'host': '127.0.0.1', 'port': 7497}
    manager = IBConnectionManager(config)
    manager.ib.connectAsync = AsyncMock(side_effect=Exception("Connection failed"))
    
    start_time = asyncio.get_event_loop().time()
    result = await manager.connect()
    elapsed = asyncio.get_event_loop().time() - start_time
    
    assert result is False
    assert elapsed >= 6  # 1 + 2 + 4 seconds minimum

@pytest.mark.asyncio  
async def test_rate_limiting():
    """Test rate limiter prevents exceeding 50 msg/sec"""
    manager = IBConnectionManager({})
    
    # Send 60 requests rapidly
    start = asyncio.get_event_loop().time()
    for _ in range(60):
        await manager._rate_limit()
    elapsed = asyncio.get_event_loop().time() - start
    
    # Should take at least 1 second for 60 requests (50/sec limit)
    assert elapsed >= 1.0

def test_criteria_validation():
    """Test criteria validation catches invalid ranges"""
    from src.scanner.criteria import PriceCriteria
    
    # Valid criteria
    valid = PriceCriteria(min_price=10, max_price=100)
    assert valid.min_price == 10
    
    # Invalid criteria
    with pytest.raises(ValueError):
        PriceCriteria(min_price=100, max_price=10)
```

```bash
# Run unit tests
cd /home/jelcke/dev/test/stockscanner
uv run pytest tests/ -v --asyncio-mode=auto

# If failing: Read error, fix code, re-run
```

### Level 3: Integration Test
```bash
# Start the scanner in demo mode
uv run python -m src.main scan --demo --config config.yaml

# Test IB connection (requires IB Gateway running)
uv run python -m src.main test-connection

# Run a quick scan
uv run python -m src.main scan --symbols AAPL,MSFT,GOOGL --duration 60

# Expected output:
# Connected to IB Gateway
# Scanning 3 symbols...
# Results displayed in rich table format
```

### Level 4: Performance Test
```bash
# Test with larger symbol list
uv run python -m src.main scan --universe SP500 --duration 300

# Monitor:
# - Memory usage stays stable
# - No rate limit errors
# - All symbols get data
```

## Final Validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] IB connection test successful
- [ ] Demo mode scan works
- [ ] Live scan with 10 symbols works
- [ ] Alerts fire correctly
- [ ] No memory leaks after 5 min run
- [ ] Logs are informative at INFO level
- [ ] All credentials in .env, not code

## Anti-Patterns to Avoid
- ❌ Don't create new IB connections per symbol - reuse one connection
- ❌ Don't skip symbol qualification - will cause subscription failures
- ❌ Don't ignore rate limits - IB will disconnect you
- ❌ Don't use sync operations in async context - will block event loop
- ❌ Don't store passwords in code - use .env and python-dotenv
- ❌ Don't let memory grow unbounded - use circular buffers
- ❌ Don't query DB in hot loops - use caching
- ❌ Don't trust external data - validate everything

## Confidence Score: 9/10

This PRP provides comprehensive context for one-pass implementation with:
- Complete documentation references and gotchas
- Detailed task breakdown with specific patterns to follow
- Working code examples from the existing codebase
- Clear validation steps at multiple levels
- All critical IB API quirks documented

The AI agent should be able to implement a working scanner by following this PRP step-by-step.