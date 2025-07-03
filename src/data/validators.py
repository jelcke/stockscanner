"""
Data validation utilities for ensuring data integrity
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..config.constants import MAX_PRICE, MAX_VOLUME, MIN_PRICE

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate and clean stock market data"""

    @staticmethod
    def validate_ohlcv(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Validate and clean OHLCV data.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            Tuple of (cleaned DataFrame, list of validation warnings)
        """
        warnings = []
        required_columns = ["open", "high", "low", "close", "volume"]

        # Check required columns
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create a copy to avoid modifying original
        clean_data = data.copy()

        # Remove duplicates based on index (assuming datetime index)
        if clean_data.index.duplicated().any():
            duplicates = clean_data.index.duplicated().sum()
            warnings.append(f"Removed {duplicates} duplicate timestamps")
            clean_data = clean_data[~clean_data.index.duplicated(keep="first")]

        # Sort by index (timestamp)
        if not clean_data.index.is_monotonic_increasing:
            warnings.append("Data was not sorted by timestamp, now sorted")
            clean_data = clean_data.sort_index()

        # Validate OHLC relationships
        invalid_high_low = clean_data["high"] < clean_data["low"]
        if invalid_high_low.any():
            count = invalid_high_low.sum()
            warnings.append(f"Fixed {count} rows where high < low")
            clean_data.loc[invalid_high_low, ["high", "low"]] = clean_data.loc[
                invalid_high_low, ["low", "high"]
            ].values

        # Ensure high >= open, close
        invalid_high = (clean_data["high"] < clean_data["open"]) | (
            clean_data["high"] < clean_data["close"]
        )
        if invalid_high.any():
            count = invalid_high.sum()
            warnings.append(f"Fixed {count} rows where high was not the highest price")
            clean_data.loc[invalid_high, "high"] = clean_data.loc[
                invalid_high, ["open", "close", "high"]
            ].max(axis=1)

        # Ensure low <= open, close
        invalid_low = (clean_data["low"] > clean_data["open"]) | (
            clean_data["low"] > clean_data["close"]
        )
        if invalid_low.any():
            count = invalid_low.sum()
            warnings.append(f"Fixed {count} rows where low was not the lowest price")
            clean_data.loc[invalid_low, "low"] = clean_data.loc[
                invalid_low, ["open", "close", "low"]
            ].min(axis=1)

        # Validate price ranges
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            # Check for negative prices
            negative_prices = clean_data[col] < 0
            if negative_prices.any():
                count = negative_prices.sum()
                warnings.append(f"Removed {count} negative prices in {col}")
                clean_data.loc[negative_prices, col] = np.nan

            # Check for unrealistic prices
            unrealistic_high = clean_data[col] > MAX_PRICE
            if unrealistic_high.any():
                count = unrealistic_high.sum()
                warnings.append(f"Capped {count} unrealistic high prices in {col}")
                clean_data.loc[unrealistic_high, col] = MAX_PRICE

            unrealistic_low = (clean_data[col] < MIN_PRICE) & (clean_data[col] > 0)
            if unrealistic_low.any():
                count = unrealistic_low.sum()
                warnings.append(f"Set {count} near-zero prices to minimum in {col}")
                clean_data.loc[unrealistic_low, col] = MIN_PRICE

        # Validate volume
        negative_volume = clean_data["volume"] < 0
        if negative_volume.any():
            count = negative_volume.sum()
            warnings.append(f"Set {count} negative volumes to 0")
            clean_data.loc[negative_volume, "volume"] = 0

        unrealistic_volume = clean_data["volume"] > MAX_VOLUME
        if unrealistic_volume.any():
            count = unrealistic_volume.sum()
            warnings.append(f"Capped {count} unrealistic volumes")
            clean_data.loc[unrealistic_volume, "volume"] = MAX_VOLUME

        # Remove rows with all NaN prices
        all_nan_prices = clean_data[price_cols].isna().all(axis=1)
        if all_nan_prices.any():
            count = all_nan_prices.sum()
            warnings.append(f"Removed {count} rows with all NaN prices")
            clean_data = clean_data[~all_nan_prices]

        # Forward fill NaN values in prices (but not volume)
        nan_count = clean_data[price_cols].isna().sum().sum()
        if nan_count > 0:
            warnings.append(f"Forward-filled {nan_count} NaN price values")
            clean_data[price_cols] = clean_data[price_cols].fillna(method="ffill")
            # If still NaN (at the beginning), backward fill
            clean_data[price_cols] = clean_data[price_cols].fillna(method="bfill")

        # Log validation summary
        if warnings:
            logger.warning(f"Data validation warnings: {'; '.join(warnings)}")
        else:
            logger.info("Data passed all validation checks")

        return clean_data, warnings

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """
        Validate stock symbol format.

        Args:
            symbol: Stock symbol to validate

        Returns:
            True if valid, False otherwise
        """
        if not symbol:
            return False

        # Basic validation rules
        if not 1 <= len(symbol) <= 10:
            return False

        # Only allow letters, numbers, dots, and hyphens
        if not all(c.isalnum() or c in ".-" for c in symbol):
            return False

        # Must start with a letter
        if not symbol[0].isalpha():
            return False

        return True

    @staticmethod
    def validate_price_change(
        current: float, previous: float, max_change_pct: float = 50.0
    ) -> bool:
        """
        Validate price change is within reasonable bounds.

        Args:
            current: Current price
            previous: Previous price
            max_change_pct: Maximum allowed change percentage

        Returns:
            True if valid, False otherwise
        """
        if previous <= 0:
            return False

        change_pct = abs((current - previous) / previous * 100)
        return change_pct <= max_change_pct

    @staticmethod
    def validate_volume_surge(
        current_volume: int, avg_volume: float, max_surge: float = 100.0
    ) -> bool:
        """
        Validate volume surge is within reasonable bounds.

        Args:
            current_volume: Current volume
            avg_volume: Average volume
            max_surge: Maximum allowed surge multiplier

        Returns:
            True if valid, False otherwise
        """
        if avg_volume <= 0:
            return True  # Can't validate without average

        surge_ratio = current_volume / avg_volume
        return surge_ratio <= max_surge

    @staticmethod
    def detect_outliers(data: pd.Series, method: str = "iqr", threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers in a data series.

        Args:
            data: Series to check for outliers
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            Boolean series marking outliers
        """
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)

        elif method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > threshold

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    @staticmethod
    def validate_ticker_data(ticker_data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate real-time ticker data from IB.

        Args:
            ticker_data: Dictionary with ticker data

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check required fields
        required_fields = ["symbol", "last", "volume"]
        for field in required_fields:
            if field not in ticker_data or ticker_data[field] is None:
                issues.append(f"Missing required field: {field}")

        if issues:
            return False, issues

        # Validate price
        last_price = ticker_data.get("last", 0)
        if last_price <= 0:
            issues.append("Invalid last price <= 0")
        elif last_price > MAX_PRICE:
            issues.append(f"Price {last_price} exceeds maximum {MAX_PRICE}")

        # Validate volume
        volume = ticker_data.get("volume", 0)
        if volume < 0:
            issues.append("Negative volume")
        elif volume > MAX_VOLUME:
            issues.append(f"Volume {volume} exceeds maximum {MAX_VOLUME}")

        # Validate bid/ask spread if available
        bid = ticker_data.get("bid")
        ask = ticker_data.get("ask")
        if bid and ask and bid > 0 and ask > 0:
            spread_pct = (ask - bid) / bid * 100
            if spread_pct > 10:  # More than 10% spread is suspicious
                issues.append(f"Excessive bid/ask spread: {spread_pct:.2f}%")

        # Validate high/low if available
        high = ticker_data.get("high")
        low = ticker_data.get("low")
        if high and low:
            if high < low:
                issues.append("High price less than low price")
            if last_price > high * 1.1:  # Last price shouldn't be much above high
                issues.append("Last price significantly above daily high")
            if last_price < low * 0.9:  # Last price shouldn't be much below low
                issues.append("Last price significantly below daily low")

        return len(issues) == 0, issues

    @staticmethod
    def clean_dataframe(
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        handle_missing: str = "forward_fill",
        remove_outliers: bool = False,
    ) -> pd.DataFrame:
        """
        General purpose dataframe cleaning.

        Args:
            df: DataFrame to clean
            drop_duplicates: Whether to drop duplicate rows
            handle_missing: How to handle missing values ('drop', 'forward_fill', 'interpolate')
            remove_outliers: Whether to remove outliers

        Returns:
            Cleaned DataFrame
        """
        clean_df = df.copy()

        # Drop duplicates
        if drop_duplicates and clean_df.duplicated().any():
            before = len(clean_df)
            clean_df = clean_df.drop_duplicates()
            logger.info(f"Dropped {before - len(clean_df)} duplicate rows")

        # Handle missing values
        if handle_missing == "drop":
            before = len(clean_df)
            clean_df = clean_df.dropna()
            logger.info(f"Dropped {before - len(clean_df)} rows with missing values")
        elif handle_missing == "forward_fill":
            clean_df = clean_df.fillna(method="ffill")
        elif handle_missing == "interpolate":
            clean_df = clean_df.interpolate(method="linear")

        # Remove outliers
        if remove_outliers:
            numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                outliers = DataValidator.detect_outliers(clean_df[col])
                if outliers.any():
                    logger.info(f"Removing {outliers.sum()} outliers from {col}")
                    clean_df = clean_df[~outliers]

        return clean_df
