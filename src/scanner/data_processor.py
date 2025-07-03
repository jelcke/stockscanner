"""
Real-time data processing - handles ticker updates and calculates indicators
"""

import logging
from collections import deque
from typing import Any, Optional

import pandas as pd
from ib_async import BarData, Ticker

from ..config.constants import (
    DEFAULT_MACD_FAST,
    DEFAULT_MACD_SIGNAL,
    DEFAULT_MACD_SLOW,
    DEFAULT_RSI_PERIOD,
    DEFAULT_SMA_PERIODS,
)
from ..data.cache_manager import CacheManager
from ..data.database import Database
from ..data.validators import DataValidator
from .scanner import ScannerResult

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators from price/volume data"""

    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=period).mean()

    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(prices: pd.Series, period: int = DEFAULT_RSI_PERIOD) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(
        prices: pd.Series,
        fast: int = DEFAULT_MACD_FAST,
        slow: int = DEFAULT_MACD_SLOW,
        signal: int = DEFAULT_MACD_SIGNAL,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD - Moving Average Convergence Divergence"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        prices: pd.Series, period: int = 20, std_dev: int = 2
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return upper_band, sma, lower_band

    @staticmethod
    def volume_analysis(volumes: pd.Series, period: int = 20) -> dict[str, float]:
        """Volume analysis metrics"""
        avg_volume = volumes.rolling(window=period).mean()
        current_volume = volumes.iloc[-1] if len(volumes) > 0 else 0
        avg_vol_value = avg_volume.iloc[-1] if len(avg_volume) > 0 else 1

        volume_ratio = current_volume / avg_vol_value if avg_vol_value > 0 else 0
        volume_surge = volume_ratio > 2.0  # Volume more than 2x average

        return {
            "current_volume": current_volume,
            "avg_volume": avg_vol_value,
            "volume_ratio": volume_ratio,
            "volume_surge": volume_surge,
        }

    @staticmethod
    def support_resistance(highs: pd.Series, lows: pd.Series, period: int = 20) -> dict[str, float]:
        """Calculate support and resistance levels"""
        # Simple approach: use rolling max/min
        resistance = highs.rolling(window=period).max().iloc[-1]
        support = lows.rolling(window=period).min().iloc[-1]

        # Pivot points
        last_high = highs.iloc[-1]
        last_low = lows.iloc[-1]
        last_close = (last_high + last_low) / 2  # Approximate

        pivot = (last_high + last_low + last_close) / 3
        r1 = 2 * pivot - last_low
        s1 = 2 * pivot - last_high

        return {"resistance": resistance, "support": support, "pivot": pivot, "r1": r1, "s1": s1}


class DataProcessor:
    """Process real-time market data and calculate metrics"""

    def __init__(self, database: Database, cache: CacheManager):
        self.database = database
        self.cache = cache
        self.validator = DataValidator()
        self.indicators = TechnicalIndicators()

        # Store recent data for calculations
        self._price_history: dict[str, deque] = {}
        self._volume_history: dict[str, deque] = {}
        self._max_history = 200  # Keep last 200 data points

    async def process_ticker(self, ticker: Ticker) -> Optional[ScannerResult]:
        """
        Process ticker data and create scan result.

        Args:
            ticker: IB ticker object

        Returns:
            ScannerResult if successful, None otherwise
        """
        try:
            symbol = ticker.contract.symbol

            # Validate ticker data
            ticker_dict = {
                "symbol": symbol,
                "last": ticker.last,
                "volume": ticker.volume,
                "bid": ticker.bid,
                "ask": ticker.ask,
                "high": ticker.high,
                "low": ticker.low,
            }

            is_valid, issues = self.validator.validate_ticker_data(ticker_dict)
            if not is_valid:
                logger.warning(f"Invalid ticker data for {symbol}: {issues}")
                return None

            # Get previous close for change calculation
            prev_close = await self._get_previous_close(symbol)
            if not prev_close:
                logger.warning(f"No previous close for {symbol}")
                return None

            # Calculate basic metrics
            change_pct = ((ticker.last - prev_close) / prev_close) * 100

            # Update price/volume history
            self._update_history(symbol, ticker.last, ticker.volume)

            # Get historical data for technical indicators
            hist_data = await self._get_or_fetch_historical_data(symbol)

            # Calculate technical indicators if we have enough data
            technical_signals = {}
            if hist_data is not None and len(hist_data) >= 50:
                technical_signals = self._calculate_technical_indicators(hist_data)

            # Calculate volume metrics
            volume_metrics = self._calculate_volume_metrics(symbol, ticker.volume)

            # Create scan result
            result = ScannerResult(
                symbol=symbol,
                price=ticker.last,
                volume=ticker.volume,
                change_pct=change_pct,
                volume_ratio=volume_metrics.get("volume_ratio", 1.0),
                technical_signals=technical_signals,
                metadata={
                    "bid": ticker.bid,
                    "ask": ticker.ask,
                    "high": ticker.high,
                    "low": ticker.low,
                    "prev_close": prev_close,
                    "spread": ticker.ask - ticker.bid if ticker.bid and ticker.ask else None,
                    "volume_surge": volume_metrics.get("volume_surge", False),
                },
            )

            # Save price data to database
            await self._save_price_data(result)

            return result

        except Exception as e:
            logger.error(f"Error processing ticker for {ticker.contract.symbol}: {e}")
            return None

    def _update_history(self, symbol: str, price: float, volume: int):
        """Update price and volume history for a symbol"""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._max_history)
            self._volume_history[symbol] = deque(maxlen=self._max_history)

        self._price_history[symbol].append(price)
        self._volume_history[symbol].append(volume)

    async def _get_previous_close(self, symbol: str) -> Optional[float]:
        """Get previous close price for a symbol"""
        # Try cache first
        cached_close = await self.cache.get(f"prev_close:{symbol}")
        if cached_close:
            return float(cached_close)

        # Query database
        latest_price = self.database.get_latest_price(symbol)
        if latest_price:
            prev_close = latest_price.close
            # Cache for future use
            await self.cache.set(f"prev_close:{symbol}", prev_close, ttl=3600)  # 1 hour
            return prev_close

        return None

    async def _get_or_fetch_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical data from cache or database"""
        # Try cache first
        cached_data = await self.cache.get(f"hist_data:{symbol}")
        if cached_data:
            return cached_data

        # Get from database
        price_history = self.database.get_price_history(symbol, days=30)

        if not price_history:
            return None

        # Convert to DataFrame
        data = pd.DataFrame(
            [
                {
                    "timestamp": p.timestamp,
                    "open": p.open,
                    "high": p.high,
                    "low": p.low,
                    "close": p.close,
                    "volume": p.volume,
                }
                for p in price_history
            ]
        )

        if data.empty:
            return None

        data.set_index("timestamp", inplace=True)
        data.sort_index(inplace=True)

        # Validate and clean data
        data, warnings = self.validator.validate_ohlcv(data)

        # Cache for future use
        await self.cache.set(f"hist_data:{symbol}", data, ttl=300)  # 5 minutes

        return data

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate all technical indicators"""
        signals = {}

        try:
            close_prices = data["close"]

            # Moving averages
            for period in DEFAULT_SMA_PERIODS:
                if len(close_prices) >= period:
                    sma = self.indicators.sma(close_prices, period)
                    signals[f"sma_{period}"] = sma.iloc[-1]

            # RSI
            if len(close_prices) >= DEFAULT_RSI_PERIOD + 1:
                rsi = self.indicators.rsi(close_prices)
                current_rsi = rsi.iloc[-1]
                signals["rsi"] = current_rsi
                signals["rsi_signal"] = (
                    "oversold"
                    if current_rsi < 30
                    else "overbought" if current_rsi > 70 else "neutral"
                )

            # MACD
            if len(close_prices) >= DEFAULT_MACD_SLOW + DEFAULT_MACD_SIGNAL:
                macd, signal, histogram = self.indicators.macd(close_prices)
                signals["macd"] = macd.iloc[-1]
                signals["macd_signal"] = signal.iloc[-1]
                signals["macd_histogram"] = histogram.iloc[-1]
                signals["macd_cross"] = (
                    "bullish"
                    if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0
                    else "bearish" if histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0 else "none"
                )

            # Bollinger Bands
            if len(close_prices) >= 20:
                upper, middle, lower = self.indicators.bollinger_bands(close_prices)
                current_price = close_prices.iloc[-1]
                signals["bb_upper"] = upper.iloc[-1]
                signals["bb_middle"] = middle.iloc[-1]
                signals["bb_lower"] = lower.iloc[-1]
                signals["bb_position"] = (
                    "above"
                    if current_price > upper.iloc[-1]
                    else "below" if current_price < lower.iloc[-1] else "inside"
                )

            # Support/Resistance
            if len(data) >= 20:
                sr_levels = self.indicators.support_resistance(data["high"], data["low"])
                signals.update(sr_levels)

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")

        return signals

    def _calculate_volume_metrics(self, symbol: str, current_volume: int) -> dict[str, Any]:
        """Calculate volume-based metrics"""
        if symbol not in self._volume_history or len(self._volume_history[symbol]) < 20:
            return {"volume_ratio": 1.0, "volume_surge": False}

        volumes = pd.Series(list(self._volume_history[symbol]))
        return self.indicators.volume_analysis(volumes)

    async def _save_price_data(self, result: ScannerResult):
        """Save price data to database"""
        try:
            price_data = {
                "symbol": result.symbol,
                "timestamp": result.timestamp,
                "open": result.metadata.get("open", result.price),
                "high": result.metadata.get("high", result.price),
                "low": result.metadata.get("low", result.price),
                "close": result.price,
                "volume": result.volume,
            }

            self.database.save_price_data([price_data])

            # Also save technical indicators if available
            if result.technical_signals:
                for indicator_name, value in result.technical_signals.items():
                    if isinstance(value, (int, float)):  # Only save numeric values
                        indicator_data = {
                            "symbol": result.symbol,
                            "timestamp": result.timestamp,
                            "indicator_name": indicator_name,
                            "value": value,
                            "signal": result.technical_signals.get(f"{indicator_name}_signal"),
                        }
                        self.database.save_indicator(indicator_data)

        except Exception as e:
            logger.error(f"Error saving price data for {result.symbol}: {e}")

    async def process_historical_bars(self, symbol: str, bars: list[BarData]) -> pd.DataFrame:
        """
        Process historical bar data from IB.

        Args:
            symbol: Stock symbol
            bars: List of BarData from IB

        Returns:
            DataFrame with processed data
        """
        if not bars:
            return pd.DataFrame()

        # Convert to DataFrame
        data = pd.DataFrame(
            [
                {
                    "timestamp": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ]
        )

        data.set_index("timestamp", inplace=True)
        data.sort_index(inplace=True)

        # Validate and clean
        data, warnings = self.validator.validate_ohlcv(data)

        # Save to database
        price_records = []
        for timestamp, row in data.iterrows():
            price_records.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
            )

        if price_records:
            self.database.save_price_data(price_records)

        return data
