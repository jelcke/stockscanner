"""
Scanner data models
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ScannerResult:
    """Result from a scan"""

    symbol: str
    price: float
    volume: int
    change_pct: float
    volume_ratio: float
    criteria_matched: list[str] = field(default_factory=list)
    technical_signals: dict[str, Any] = field(default_factory=dict)
    ml_predictions: dict[str, float] = field(default_factory=dict)
    sentiment_score: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    extra_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "change_pct": self.change_pct,
            "volume_ratio": self.volume_ratio,
            "criteria_matched": ", ".join(self.criteria_matched),
            "breakout_probability": self.ml_predictions.get("breakout", 0.0),
            "sentiment_score": self.sentiment_score,
            "scan_time": self.timestamp,
            "extra_data": self.extra_data,
        }