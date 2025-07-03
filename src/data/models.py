"""
SQLAlchemy data models for the stock scanner
"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Index, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Stock(Base):
    """Stock information"""

    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    company_name = Column(String(100))
    sector = Column(String(50))
    exchange = Column(String(20))
    market_cap = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Stock(symbol='{self.symbol}', company='{self.company_name}')>"


class PriceData(Base):
    """Historical and real-time price data"""

    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    vwap = Column(Float)  # Volume weighted average price

    # Create composite index for efficient queries
    __table_args__ = (Index("idx_symbol_timestamp", "symbol", "timestamp"),)

    def __repr__(self):
        return f"<PriceData(symbol='{self.symbol}', time='{self.timestamp}', close={self.close})>"


class ScanResult(Base):
    """Results from scanner runs"""

    __tablename__ = "scan_results"

    id = Column(Integer, primary_key=True)
    scan_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    symbol = Column(String(10), nullable=False)
    criteria_matched = Column(String(200))
    price = Column(Float)
    volume = Column(Integer)
    change_pct = Column(Float)
    volume_ratio = Column(Float)  # Current volume / avg volume
    breakout_probability = Column(Float)
    sentiment_score = Column(Float)
    alert_sent = Column(Boolean, default=False)
    extra_data = Column(JSON)  # Additional data as JSON

    def __repr__(self):
        return f"<ScanResult(symbol='{self.symbol}', time='{self.scan_time}', change={self.change_pct}%)>"


class Alert(Base):
    """Alert history"""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    symbol = Column(String(10), nullable=False)
    alert_type = Column(String(50))  # 'price_surge', 'volume_spike', 'breakout', etc.
    alert_level = Column(String(20))  # 'info', 'warning', 'critical'
    message = Column(String(500))
    channels = Column(String(100))  # Comma-separated list of channels
    sent_successfully = Column(Boolean, default=False)
    error_message = Column(String(200))

    def __repr__(self):
        return (
            f"<Alert(symbol='{self.symbol}', type='{self.alert_type}', time='{self.created_at}')>"
        )


class ModelPrediction(Base):
    """ML model predictions history"""

    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True)
    prediction_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    symbol = Column(String(10), nullable=False)
    model_name = Column(String(50))  # 'breakout_rf', 'pattern_cnn', etc.
    model_version = Column(String(20))
    prediction_type = Column(String(50))  # 'breakout', 'pattern', etc.
    prediction_value = Column(Float)  # Probability or score
    confidence = Column(Float)
    features_used = Column(JSON)  # Feature values used for prediction
    actual_outcome = Column(Boolean)  # For backtesting/validation

    __table_args__ = (Index("idx_symbol_prediction_time", "symbol", "prediction_time"),)

    def __repr__(self):
        return f"<ModelPrediction(symbol='{self.symbol}', model='{self.model_name}', value={self.prediction_value})>"


class SentimentData(Base):
    """Sentiment analysis results"""

    __tablename__ = "sentiment_data"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    symbol = Column(String(10), nullable=False)
    source = Column(String(50))  # 'news', 'reddit', 'twitter', etc.
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # 'positive', 'negative', 'neutral'
    text_sample = Column(String(500))  # Sample text
    url = Column(String(500))

    __table_args__ = (Index("idx_symbol_timestamp_sentiment", "symbol", "timestamp"),)

    def __repr__(self):
        return f"<SentimentData(symbol='{self.symbol}', source='{self.source}', score={self.sentiment_score})>"


class TechnicalIndicator(Base):
    """Calculated technical indicators"""

    __tablename__ = "technical_indicators"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    indicator_name = Column(String(50))  # 'RSI', 'MACD', 'SMA_20', etc.
    value = Column(Float)
    signal = Column(String(20))  # 'buy', 'sell', 'neutral'

    __table_args__ = (Index("idx_symbol_indicator_time", "symbol", "indicator_name", "timestamp"),)

    def __repr__(self):
        return f"<TechnicalIndicator(symbol='{self.symbol}', indicator='{self.indicator_name}', value={self.value})>"
