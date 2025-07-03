"""
Scanning criteria definitions with Pydantic models
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, validator

from .scanner import ScannerResult


class CriteriaType(str, Enum):
    """Types of scanning criteria"""

    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    TECHNICAL = "technical"
    PATTERN = "pattern"
    SENTIMENT = "sentiment"
    COMPOSITE = "composite"


class BaseCriteria(ABC, BaseModel):
    """Base class for all criteria"""

    name: str = Field(..., description="Criteria name")
    type: CriteriaType
    enabled: bool = Field(default=True)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)

    @abstractmethod
    def evaluate(self, result: ScannerResult) -> bool:
        """Evaluate if the result meets this criteria"""
        pass

    class Config:
        arbitrary_types_allowed = True


class PriceCriteria(BaseCriteria):
    """Price-based scanning criteria"""

    type: CriteriaType = CriteriaType.PRICE
    name: str = "Price Criteria"

    min_price: Optional[float] = Field(None, ge=0, description="Minimum stock price")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum stock price")
    min_change_pct: Optional[float] = Field(None, description="Minimum price change %")
    max_change_pct: Optional[float] = Field(None, description="Maximum price change %")

    @validator("max_price")
    def validate_price_range(cls, v, values):
        if v and values.get("min_price") and v < values["min_price"]:
            raise ValueError("max_price must be >= min_price")
        return v

    @validator("max_change_pct")
    def validate_change_range(cls, v, values):
        if v and values.get("min_change_pct") and v < values["min_change_pct"]:
            raise ValueError("max_change_pct must be >= min_change_pct")
        return v

    def evaluate(self, result: ScannerResult) -> bool:
        """Check if result meets price criteria"""
        if not self.enabled:
            return True

        # Check price range
        if self.min_price is not None and result.price < self.min_price:
            return False
        if self.max_price is not None and result.price > self.max_price:
            return False

        # Check change percentage
        if self.min_change_pct is not None and result.change_pct < self.min_change_pct:
            return False
        if self.max_change_pct is not None and result.change_pct > self.max_change_pct:
            return False

        return True


class VolumeCriteria(BaseCriteria):
    """Volume-based scanning criteria"""

    type: CriteriaType = CriteriaType.VOLUME
    name: str = "Volume Criteria"

    min_volume: Optional[int] = Field(None, ge=0, description="Minimum current volume")
    min_avg_volume: Optional[int] = Field(None, ge=0, description="Minimum average volume")
    volume_surge_multiple: Optional[float] = Field(
        None, ge=1.0, description="Volume surge multiplier"
    )

    def evaluate(self, result: ScannerResult) -> bool:
        """Check if result meets volume criteria"""
        if not self.enabled:
            return True

        # Check minimum volume
        if self.min_volume is not None and result.volume < self.min_volume:
            return False

        # Check volume surge
        if self.volume_surge_multiple is not None:
            if result.volume_ratio < self.volume_surge_multiple:
                return False

        return True


class MomentumCriteria(BaseCriteria):
    """Momentum-based scanning criteria"""

    type: CriteriaType = CriteriaType.MOMENTUM
    name: str = "Momentum Criteria"

    rsi_min: Optional[float] = Field(None, ge=0, le=100, description="Minimum RSI value")
    rsi_max: Optional[float] = Field(None, ge=0, le=100, description="Maximum RSI value")
    macd_signal: Optional[str] = Field(None, description="MACD signal (bullish/bearish)")
    consecutive_gains: Optional[int] = Field(
        None, ge=1, description="Number of consecutive gaining days"
    )

    @validator("rsi_max")
    def validate_rsi_range(cls, v, values):
        if v and values.get("rsi_min") and v < values["rsi_min"]:
            raise ValueError("rsi_max must be >= rsi_min")
        return v

    def evaluate(self, result: ScannerResult) -> bool:
        """Check if result meets momentum criteria"""
        if not self.enabled:
            return True

        signals = result.technical_signals

        # Check RSI
        if self.rsi_min is not None or self.rsi_max is not None:
            rsi = signals.get("rsi")
            if rsi is None:
                return False  # No RSI data available

            if self.rsi_min is not None and rsi < self.rsi_min:
                return False
            if self.rsi_max is not None and rsi > self.rsi_max:
                return False

        # Check MACD signal
        if self.macd_signal:
            macd_cross = signals.get("macd_cross")
            if macd_cross != self.macd_signal:
                return False

        return True


class TechnicalCriteria(BaseCriteria):
    """Technical indicator-based criteria"""

    type: CriteriaType = CriteriaType.TECHNICAL
    name: str = "Technical Criteria"

    indicator: str = Field(..., description="Indicator name (e.g., 'price', 'rsi', 'macd')")
    condition: str = Field(..., description="Condition (e.g., 'above', 'below', 'cross')")
    value: Optional[float] = Field(None, description="Comparison value")
    reference: Optional[str] = Field(None, description="Reference indicator for comparison")

    def evaluate(self, result: ScannerResult) -> bool:
        """Check if result meets technical criteria"""
        if not self.enabled:
            return True

        # Special handling for price-based conditions
        if self.indicator == "price":
            return self._evaluate_price_condition(result)

        # Get indicator value
        indicator_value = result.technical_signals.get(self.indicator)
        if indicator_value is None:
            return False  # Indicator not available

        # Evaluate condition
        if self.condition == "above":
            if self.value is not None:
                return indicator_value > self.value
            elif self.reference:
                ref_value = result.technical_signals.get(self.reference)
                return ref_value is not None and indicator_value > ref_value

        elif self.condition == "below":
            if self.value is not None:
                return indicator_value < self.value
            elif self.reference:
                ref_value = result.technical_signals.get(self.reference)
                return ref_value is not None and indicator_value < ref_value

        elif self.condition == "cross":
            # This requires historical data to detect crossovers
            # For now, check if indicator has a cross signal
            cross_signal = result.technical_signals.get(f"{self.indicator}_cross")
            return cross_signal == self.value

        return False

    def _evaluate_price_condition(self, result: ScannerResult) -> bool:
        """Evaluate price-specific conditions"""
        if self.condition == "52w_high":
            # Check if price is near 52-week high
            resistance = result.technical_signals.get("resistance")
            if resistance and self.value:
                threshold = resistance * self.value  # e.g., 0.95 for within 5% of high
                return result.price >= threshold

        elif self.condition == "above_sma":
            # Check if price is above a specific SMA
            if self.reference:
                sma_value = result.technical_signals.get(self.reference)
                return sma_value is not None and result.price > sma_value

        return False


class PatternCriteria(BaseCriteria):
    """Chart pattern-based criteria"""

    type: CriteriaType = CriteriaType.PATTERN
    name: str = "Pattern Criteria"

    patterns: list[str] = Field(default_factory=list, description="List of pattern names to match")
    min_confidence: float = Field(0.7, ge=0.0, le=1.0, description="Minimum pattern confidence")

    def evaluate(self, result: ScannerResult) -> bool:
        """Check if result has matching patterns"""
        if not self.enabled or not self.patterns:
            return True

        # Check ML predictions for pattern matches
        for pattern in self.patterns:
            confidence = result.ml_predictions.get(f"pattern_{pattern}", 0.0)
            if confidence >= self.min_confidence:
                return True

        return False


class SentimentCriteria(BaseCriteria):
    """Sentiment-based criteria"""

    type: CriteriaType = CriteriaType.SENTIMENT
    name: str = "Sentiment Criteria"

    min_sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    max_sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    sentiment_trend: Optional[str] = Field(
        None, description="Sentiment trend (improving/declining)"
    )

    def evaluate(self, result: ScannerResult) -> bool:
        """Check if result meets sentiment criteria"""
        if not self.enabled:
            return True

        if result.sentiment_score is None:
            return False  # No sentiment data

        # Check sentiment range
        if (
            self.min_sentiment_score is not None
            and result.sentiment_score < self.min_sentiment_score
        ):
            return False
        if (
            self.max_sentiment_score is not None
            and result.sentiment_score > self.max_sentiment_score
        ):
            return False

        return True


class CompositeCriteria(BaseCriteria):
    """Composite criteria combining multiple sub-criteria"""

    type: CriteriaType = CriteriaType.COMPOSITE
    name: str = "Composite Criteria"

    operator: str = Field("AND", description="Logical operator (AND/OR)")
    criteria: list[
        Union[
            PriceCriteria,
            VolumeCriteria,
            MomentumCriteria,
            TechnicalCriteria,
            PatternCriteria,
            SentimentCriteria,
        ]
    ]

    def evaluate(self, result: ScannerResult) -> bool:
        """Evaluate composite criteria"""
        if not self.enabled or not self.criteria:
            return True

        if self.operator == "AND":
            # All criteria must match
            for criterion in self.criteria:
                if not criterion.evaluate(result):
                    return False
            return True

        elif self.operator == "OR":
            # At least one criterion must match
            for criterion in self.criteria:
                if criterion.evaluate(result):
                    return True
            return False

        else:
            raise ValueError(f"Unknown operator: {self.operator}")


class CriteriaPresets:
    """Predefined criteria combinations"""

    @staticmethod
    def momentum_gainers() -> CompositeCriteria:
        """Stocks with strong momentum"""
        return CompositeCriteria(
            name="Momentum Gainers",
            operator="AND",
            criteria=[
                PriceCriteria(min_price=10, max_price=1000, min_change_pct=-10),  # Allow negative changes too
                VolumeCriteria(min_volume=10000),  # Lower minimum for testing
                # MomentumCriteria(rsi_min=50, rsi_max=80),  # Disabled - requires historical data
            ],
        )

    @staticmethod
    def volume_breakouts() -> CompositeCriteria:
        """Stocks breaking out on high volume"""
        return CompositeCriteria(
            name="Volume Breakouts",
            operator="AND",
            criteria=[
                VolumeCriteria(volume_surge_multiple=3.0),
                TechnicalCriteria(
                    indicator="price", condition="52w_high", value=0.95  # Within 5% of 52-week high
                ),
            ],
        )

    @staticmethod
    def oversold_bounce() -> CompositeCriteria:
        """Oversold stocks showing signs of reversal"""
        return CompositeCriteria(
            name="Oversold Bounce",
            operator="AND",
            criteria=[
                MomentumCriteria(rsi_max=35),
                PriceCriteria(min_change_pct=0.5),  # Starting to move up
                VolumeCriteria(volume_surge_multiple=1.2),
            ],
        )

    @staticmethod
    def technical_breakout() -> CompositeCriteria:
        """Technical breakout pattern"""
        return CompositeCriteria(
            name="Technical Breakout",
            operator="AND",
            criteria=[
                TechnicalCriteria(indicator="price", condition="above", reference="sma_50"),
                TechnicalCriteria(indicator="macd", condition="cross", value="bullish"),
                VolumeCriteria(min_volume=100000),
            ],
        )

    @staticmethod
    def high_sentiment() -> CompositeCriteria:
        """Stocks with positive sentiment"""
        return CompositeCriteria(
            name="High Sentiment",
            operator="AND",
            criteria=[
                SentimentCriteria(min_sentiment_score=0.5),
                PriceCriteria(min_price=5),
                VolumeCriteria(min_avg_volume=500000),
            ],
        )


def load_criteria_from_config(config: dict[str, Any]) -> list[BaseCriteria]:
    """
    Load criteria from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        List of criteria objects
    """
    criteria_list = []

    for criteria_config in config.get("criteria", []):
        criteria_type = criteria_config.get("type")

        if criteria_type == "price":
            criteria = PriceCriteria(**criteria_config)
        elif criteria_type == "volume":
            criteria = VolumeCriteria(**criteria_config)
        elif criteria_type == "momentum":
            criteria = MomentumCriteria(**criteria_config)
        elif criteria_type == "technical":
            criteria = TechnicalCriteria(**criteria_config)
        elif criteria_type == "pattern":
            criteria = PatternCriteria(**criteria_config)
        elif criteria_type == "sentiment":
            criteria = SentimentCriteria(**criteria_config)
        elif criteria_type == "composite":
            # Recursively load sub-criteria
            sub_criteria = load_criteria_from_config(
                {"criteria": criteria_config.get("criteria", [])}
            )
            criteria = CompositeCriteria(
                name=criteria_config.get("name", "Composite"),
                operator=criteria_config.get("operator", "AND"),
                criteria=sub_criteria,
            )
        else:
            continue

        criteria_list.append(criteria)

    return criteria_list
