"""
Database connection management and CRUD operations
"""

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import and_, create_engine, desc
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..config.constants import DB_DEFAULT_PATH
from .models import (
    Alert,
    Base,
    ModelPrediction,
    PriceData,
    ScanResult,
    SentimentData,
    Stock,
    TechnicalIndicator,
)

logger = logging.getLogger(__name__)


class Database:
    """Database connection and operations manager"""

    def __init__(self, db_path: str = None, echo: bool = False):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            echo: Whether to echo SQL statements (for debugging)
        """
        self.db_path = db_path or DB_DEFAULT_PATH

        # Create database directory if needed
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create engine with connection pooling
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,  # Better for SQLite with threading
        )

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)

        # Create tables
        self._create_tables()

    def _create_tables(self):
        """Create all tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info(f"Database tables created/verified at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    @contextmanager
    def get_session(self) -> Session:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    # Stock CRUD operations
    def get_stock(self, symbol: str) -> Optional[Stock]:
        """Get stock by symbol"""
        with self.get_session() as session:
            return session.query(Stock).filter(Stock.symbol == symbol).first()

    def create_or_update_stock(self, symbol: str, **kwargs) -> Stock:
        """Create or update stock information"""
        with self.get_session() as session:
            stock = session.query(Stock).filter(Stock.symbol == symbol).first()

            if stock:
                # Update existing
                for key, value in kwargs.items():
                    if hasattr(stock, key):
                        setattr(stock, key, value)
                stock.last_updated = datetime.utcnow()
            else:
                # Create new
                stock = Stock(symbol=symbol, **kwargs)
                session.add(stock)

            session.flush()
            session.refresh(stock)
            return stock

    def get_stocks_by_sector(self, sector: str) -> list[Stock]:
        """Get all stocks in a sector"""
        with self.get_session() as session:
            return session.query(Stock).filter(Stock.sector == sector).all()

    # Price data CRUD operations
    def save_price_data(self, price_data: list[dict[str, Any]]) -> int:
        """
        Save multiple price data records.

        Args:
            price_data: List of dicts with price data

        Returns:
            Number of records saved
        """
        count = 0
        with self.get_session() as session:
            for data in price_data:
                # Check if record already exists
                existing = (
                    session.query(PriceData)
                    .filter(
                        and_(
                            PriceData.symbol == data["symbol"],
                            PriceData.timestamp == data["timestamp"],
                        )
                    )
                    .first()
                )

                if not existing:
                    price = PriceData(**data)
                    session.add(price)
                    count += 1

        logger.info(f"Saved {count} price records")
        return count

    def get_latest_price(self, symbol: str) -> Optional[PriceData]:
        """Get latest price data for a symbol"""
        with self.get_session() as session:
            return (
                session.query(PriceData)
                .filter(PriceData.symbol == symbol)
                .order_by(desc(PriceData.timestamp))
                .first()
            )

    def get_price_history(self, symbol: str, days: int = 30) -> list[PriceData]:
        """Get price history for a symbol"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        with self.get_session() as session:
            return (
                session.query(PriceData)
                .filter(and_(PriceData.symbol == symbol, PriceData.timestamp >= cutoff_date))
                .order_by(PriceData.timestamp)
                .all()
            )

    # Scan results CRUD operations
    def save_scan_result(self, scan_result: dict[str, Any]) -> ScanResult:
        """Save a scan result"""
        with self.get_session() as session:
            result = ScanResult(**scan_result)
            session.add(result)
            session.flush()
            session.refresh(result)
            return result

    def get_recent_scan_results(self, minutes: int = 60, limit: int = 100) -> list[ScanResult]:
        """Get recent scan results"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        with self.get_session() as session:
            return (
                session.query(ScanResult)
                .filter(ScanResult.scan_time >= cutoff_time)
                .order_by(desc(ScanResult.scan_time))
                .limit(limit)
                .all()
            )

    def get_top_movers(self, limit: int = 20) -> list[ScanResult]:
        """Get top moving stocks from recent scans"""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)

        with self.get_session() as session:
            return (
                session.query(ScanResult)
                .filter(ScanResult.scan_time >= cutoff_time)
                .order_by(desc(ScanResult.change_pct.abs()))
                .limit(limit)
                .all()
            )

    # Alert CRUD operations
    def save_alert(self, alert_data: dict[str, Any]) -> Alert:
        """Save an alert"""
        with self.get_session() as session:
            alert = Alert(**alert_data)
            session.add(alert)
            session.flush()
            session.refresh(alert)
            return alert

    def get_alerts_for_symbol(self, symbol: str, days: int = 7) -> list[Alert]:
        """Get alerts for a specific symbol"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        with self.get_session() as session:
            return (
                session.query(Alert)
                .filter(and_(Alert.symbol == symbol, Alert.created_at >= cutoff_date))
                .order_by(desc(Alert.created_at))
                .all()
            )

    # Model prediction CRUD operations
    def save_prediction(self, prediction_data: dict[str, Any]) -> ModelPrediction:
        """Save a model prediction"""
        with self.get_session() as session:
            prediction = ModelPrediction(**prediction_data)
            session.add(prediction)
            session.flush()
            session.refresh(prediction)
            return prediction

    def get_predictions_for_symbol(
        self, symbol: str, model_name: str = None
    ) -> list[ModelPrediction]:
        """Get predictions for a symbol"""
        with self.get_session() as session:
            query = session.query(ModelPrediction).filter(ModelPrediction.symbol == symbol)

            if model_name:
                query = query.filter(ModelPrediction.model_name == model_name)

            return query.order_by(desc(ModelPrediction.prediction_time)).all()

    # Sentiment data CRUD operations
    def save_sentiment(self, sentiment_data: dict[str, Any]) -> SentimentData:
        """Save sentiment data"""
        with self.get_session() as session:
            sentiment = SentimentData(**sentiment_data)
            session.add(sentiment)
            session.flush()
            session.refresh(sentiment)
            return sentiment

    def get_sentiment_summary(self, symbol: str, hours: int = 24) -> dict[str, float]:
        """Get sentiment summary for a symbol"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        with self.get_session() as session:
            sentiments = (
                session.query(SentimentData)
                .filter(
                    and_(SentimentData.symbol == symbol, SentimentData.timestamp >= cutoff_time)
                )
                .all()
            )

        if not sentiments:
            return {"average": 0.0, "count": 0}

        scores = [s.sentiment_score for s in sentiments]
        return {
            "average": sum(scores) / len(scores),
            "count": len(scores),
            "positive": len([s for s in scores if s > 0.1]),
            "negative": len([s for s in scores if s < -0.1]),
            "neutral": len([s for s in scores if -0.1 <= s <= 0.1]),
        }

    # Technical indicators CRUD operations
    def save_indicator(self, indicator_data: dict[str, Any]) -> TechnicalIndicator:
        """Save technical indicator value"""
        with self.get_session() as session:
            indicator = TechnicalIndicator(**indicator_data)
            session.add(indicator)
            session.flush()
            session.refresh(indicator)
            return indicator

    def get_latest_indicators(self, symbol: str) -> dict[str, Any]:
        """Get latest values for all indicators for a symbol"""
        with self.get_session() as session:
            # Get distinct indicator names
            indicators = session.query(TechnicalIndicator.indicator_name).distinct().all()

            result = {}
            for (indicator_name,) in indicators:
                latest = (
                    session.query(TechnicalIndicator)
                    .filter(
                        and_(
                            TechnicalIndicator.symbol == symbol,
                            TechnicalIndicator.indicator_name == indicator_name,
                        )
                    )
                    .order_by(desc(TechnicalIndicator.timestamp))
                    .first()
                )

                if latest:
                    result[indicator_name] = {
                        "value": latest.value,
                        "signal": latest.signal,
                        "timestamp": latest.timestamp,
                    }

        return result

    # Utility methods
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data to manage database size"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        with self.get_session() as session:
            # Delete old price data
            deleted_prices = (
                session.query(PriceData).filter(PriceData.timestamp < cutoff_date).delete()
            )

            # Delete old scan results
            deleted_scans = (
                session.query(ScanResult).filter(ScanResult.scan_time < cutoff_date).delete()
            )

            # Delete old alerts
            deleted_alerts = session.query(Alert).filter(Alert.created_at < cutoff_date).delete()

        logger.info(
            f"Cleaned up old data: {deleted_prices} prices, {deleted_scans} scans, {deleted_alerts} alerts"
        )

    def get_database_stats(self) -> dict[str, int]:
        """Get database statistics"""
        with self.get_session() as session:
            return {
                "stocks": session.query(Stock).count(),
                "price_records": session.query(PriceData).count(),
                "scan_results": session.query(ScanResult).count(),
                "alerts": session.query(Alert).count(),
                "predictions": session.query(ModelPrediction).count(),
                "sentiments": session.query(SentimentData).count(),
                "indicators": session.query(TechnicalIndicator).count(),
            }
