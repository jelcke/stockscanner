# Stock Scanner Project Planning

## Project Overview
Interactive Brokers Real-Time Stock Scanner with AI-Enhanced Pattern Recognition

## Architecture

### Core Components
1. **Scanner Core** (`src/scanner/`)
   - `scanner.py` - Main scanner logic
   - `ib_connection.py` - IB API connection management
   - `data_processor.py` - Real-time data processing
   - `criteria.py` - Scanning criteria definitions

2. **AI/ML Module** (`src/ml/`)
   - `pattern_detector.py` - Chart pattern recognition
   - `breakout_predictor.py` - ML breakout prediction
   - `feature_engineering.py` - Feature extraction
   - `model_manager.py` - Model loading/inference

3. **Sentiment Analysis** (`src/sentiment/`)
   - `news_analyzer.py` - News sentiment
   - `social_analyzer.py` - Social media sentiment
   - `sentiment_aggregator.py` - Combined sentiment scores

4. **Data Management** (`src/data/`)
   - `cache_manager.py` - Redis caching
   - `database.py` - SQLite persistence
   - `models.py` - Data models (SQLAlchemy)
   - `validators.py` - Data validation

5. **Alerts & Display** (`src/alerts/`)
   - `alert_manager.py` - Alert orchestration
   - `channels.py` - Email/Telegram/Webhook
   - `display.py` - Rich terminal UI

6. **Configuration** (`src/config/`)
   - `settings.py` - Configuration management
   - `constants.py` - System constants

## Technology Stack
- **Core**: Python 3.9+, asyncio
- **IB API**: ib_async
- **ML**: scikit-learn, tensorflow, opencv-python
- **Data**: pandas, numpy, redis, sqlite3
- **Display**: rich
- **Testing**: pytest, pytest-asyncio

## Style Guidelines
- PEP8 compliance with black formatter
- Type hints everywhere
- Google-style docstrings
- Async/await for IB operations
- Dataclasses for data models
- Comprehensive error handling

## File Naming Conventions
- Snake_case for modules and functions
- PascalCase for classes
- No files over 500 lines
- Clear module separation by feature

## Testing Strategy
- Unit tests for all components
- Integration tests with mock IB API
- Performance tests for real-time processing
- Backtesting validation

## Critical Constraints
- IB API rate limit: 50 msg/sec
- Market data lines: 100 standard
- Memory management for streams
- Thread safety for callbacks
- Security for credentials