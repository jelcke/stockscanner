"""
Core scanner orchestration - manages scanning process and coordinates components
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from ib_async import Ticker

from ..config.constants import DEFAULT_SCAN_INTERVAL
from ..config.settings import get_config
from ..data.cache_manager import CacheManager
from ..data.database import Database
from .data_processor import DataProcessor
from .ib_connection import IBConnectionManager

logger = logging.getLogger(__name__)


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
    metadata: dict[str, Any] = field(default_factory=dict)

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
            "metadata": self.metadata,
        }


class StockScanner:
    """Main scanner class that coordinates all scanning operations"""

    def __init__(
        self,
        ib_manager: Optional[IBConnectionManager] = None,
        database: Optional[Database] = None,
        cache: Optional[CacheManager] = None,
    ):
        """
        Initialize scanner.

        Args:
            ib_manager: IB connection manager instance
            database: Database instance
            cache: Cache manager instance
        """
        self.config = get_config()
        self.ib_manager = ib_manager or IBConnectionManager()
        self.database = database or Database()
        self.cache = cache or CacheManager()
        self.data_processor = DataProcessor(self.database, self.cache)

        # Scanner state
        self._scanning = False
        self._scan_task: Optional[asyncio.Task] = None
        self._active_tickers: dict[str, Ticker] = {}
        self._scan_results: list[ScannerResult] = []

        # Callbacks
        self._result_callbacks: list[Callable[[ScannerResult], None]] = []

    async def connect(self) -> bool:
        """Connect to IB Gateway"""
        return await self.ib_manager.connect()

    async def disconnect(self):
        """Disconnect from IB Gateway and cleanup"""
        self._scanning = False

        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

        await self.ib_manager.disconnect()

    def add_result_callback(self, callback: Callable[[ScannerResult], None]):
        """Add callback for scan results"""
        self._result_callbacks.append(callback)

    async def scan_symbols(
        self,
        symbols: list[str],
        criteria: Optional[list[Any]] = None,
        continuous: bool = False,
        interval: int = DEFAULT_SCAN_INTERVAL,
    ) -> list[ScannerResult]:
        """
        Scan a list of symbols.

        Args:
            symbols: List of stock symbols to scan
            criteria: List of criteria objects to evaluate
            continuous: Whether to scan continuously
            interval: Scan interval in seconds (for continuous mode)

        Returns:
            List of scan results
        """
        if not await self.ib_manager.ensure_connected():
            raise ConnectionError("Failed to connect to IB Gateway")

        self._scanning = True
        self._scan_results.clear()

        if continuous:
            self._scan_task = asyncio.create_task(
                self._continuous_scan(symbols, criteria, interval)
            )
            # Return empty list for continuous mode (results via callbacks)
            return []
        else:
            # One-time scan
            return await self._scan_once(symbols, criteria)

    async def stop_scanning(self):
        """Stop continuous scanning"""
        self._scanning = False

        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

    async def _continuous_scan(
        self, symbols: list[str], criteria: Optional[list[Any]], interval: int
    ):
        """Run continuous scanning loop"""
        logger.info(f"Starting continuous scan of {len(symbols)} symbols with {interval}s interval")

        while self._scanning:
            try:
                scan_start = asyncio.get_event_loop().time()

                # Perform scan
                results = await self._scan_once(symbols, criteria)

                # Process results
                for result in results:
                    # Trigger callbacks
                    for callback in self._result_callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"Error in result callback: {e}")

                # Calculate sleep time to maintain interval
                scan_duration = asyncio.get_event_loop().time() - scan_start
                sleep_time = max(0, interval - scan_duration)

                if sleep_time > 0:
                    logger.debug(
                        f"Scan completed in {scan_duration:.1f}s, sleeping for {sleep_time:.1f}s"
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(
                        f"Scan took {scan_duration:.1f}s, longer than interval {interval}s"
                    )

            except Exception as e:
                logger.error(f"Error in continuous scan: {e}")
                await asyncio.sleep(interval)  # Still wait on error

    async def _scan_once(
        self, symbols: list[str], criteria: Optional[list[Any]]
    ) -> list[ScannerResult]:
        """Perform one scan of all symbols"""
        logger.info(f"Scanning {len(symbols)} symbols...")

        # Subscribe to market data for all symbols
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._process_symbol(symbol))
            tasks.append(task)

        # Wait for all symbols to be processed
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors and None results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing {symbols[i]}: {result}")
            elif result is not None:
                valid_results.append(result)

        # Apply criteria filtering if provided
        if criteria:
            filtered_results = []
            for result in valid_results:
                if await self._evaluate_criteria(result, criteria):
                    filtered_results.append(result)
            valid_results = filtered_results

        # Save to database
        for result in valid_results:
            try:
                self.database.save_scan_result(result.to_dict())
            except Exception as e:
                logger.error(f"Error saving result for {result.symbol}: {e}")

        # Sort by change percentage (absolute value)
        valid_results.sort(key=lambda x: abs(x.change_pct), reverse=True)

        # Limit results
        max_results = self.config.scanner.max_results
        if len(valid_results) > max_results:
            valid_results = valid_results[:max_results]

        logger.info(f"Scan complete: {len(valid_results)} results")
        return valid_results

    async def _process_symbol(self, symbol: str) -> Optional[ScannerResult]:
        """Process a single symbol"""
        try:
            # Check cache first
            cached_result = await self.cache.get_scan_result(symbol)
            if cached_result and (datetime.utcnow() - cached_result.timestamp).seconds < 30:
                logger.debug(f"Using cached result for {symbol}")
                return cached_result

            # Subscribe to market data
            ticker = await self.ib_manager.subscribe_market_data(symbol)
            if not ticker:
                return None

            # Wait for data to populate (with timeout)
            await self._wait_for_ticker_data(ticker, timeout=5.0)

            # Process the ticker data
            result = await self.data_processor.process_ticker(ticker)

            if result:
                # Cache the result
                await self.cache.set_scan_result(symbol, result)

            return result

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None

    async def _wait_for_ticker_data(self, ticker: Ticker, timeout: float = 5.0):
        """Wait for ticker data to be populated"""
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if ticker.last and ticker.volume:
                return  # Data is ready
            await asyncio.sleep(0.1)

        logger.warning(f"Timeout waiting for data for {ticker.contract.symbol}")

    async def _evaluate_criteria(self, result: ScannerResult, criteria: list[Any]) -> bool:
        """
        Evaluate if a result matches the criteria.

        Args:
            result: Scan result to evaluate
            criteria: List of criteria objects

        Returns:
            True if all criteria match
        """
        if not criteria:
            return True

        matched_criteria = []

        for criterion in criteria:
            # Each criterion should have an evaluate method
            if hasattr(criterion, "evaluate"):
                if criterion.evaluate(result):
                    matched_criteria.append(criterion.name)
                else:
                    return False  # All criteria must match

        result.criteria_matched = matched_criteria
        return True

    async def get_universe_symbols(self, universe: str = "US_STOCKS") -> list[str]:
        """
        Get list of symbols for a universe.

        Args:
            universe: Universe name (e.g., 'US_STOCKS', 'SP500', 'NASDAQ100')

        Returns:
            List of symbols
        """
        # For now, return a predefined list
        # In production, this would query a database or API
        universes = {
            "US_STOCKS": [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "JPM",
                "JNJ",
                "V",
                "PG",
                "UNH",
                "HD",
                "MA",
                "DIS",
            ],
            "TECH": [
                "AAPL",
                "MSFT",
                "GOOGL",
                "META",
                "NVDA",
                "TSLA",
                "ORCL",
                "CRM",
                "ADBE",
                "NFLX",
                "PYPL",
                "INTC",
                "AMD",
                "QCOM",
            ],
            "TEST": ["AAPL", "MSFT", "GOOGL"],  # Small set for testing
        }

        return universes.get(universe, universes["TEST"])

    def get_recent_results(self) -> list[ScannerResult]:
        """Get recent scan results from memory"""
        return self._scan_results.copy()

    async def backtest_criteria(
        self, symbols: list[str], criteria: list[Any], days: int = 30
    ) -> dict[str, Any]:
        """
        Backtest criteria on historical data.

        Args:
            symbols: List of symbols to test
            criteria: Criteria to evaluate
            days: Number of days to look back

        Returns:
            Backtest results including success rate, average returns, etc.
        """
        logger.info(f"Backtesting criteria on {len(symbols)} symbols over {days} days")

        results = {
            "total_signals": 0,
            "successful_signals": 0,
            "average_return": 0.0,
            "best_performer": None,
            "worst_performer": None,
            "symbol_results": {},
        }

        for symbol in symbols:
            try:
                # Get historical data
                bars = await self.ib_manager.get_historical_data(
                    symbol, duration_str=f"{days} D", bar_size_setting="1 day"
                )

                if not bars:
                    continue

                # Process historical data and evaluate criteria
                # This is a simplified version - real backtesting would be more complex
                symbol_signals = 0
                symbol_returns = []

                for i in range(1, len(bars)):
                    # Create a mock result from historical data
                    mock_result = ScannerResult(
                        symbol=symbol,
                        price=bars[i].close,
                        volume=bars[i].volume,
                        change_pct=((bars[i].close - bars[i - 1].close) / bars[i - 1].close) * 100,
                        volume_ratio=1.0,  # Would need average volume for real ratio
                    )

                    # Evaluate criteria
                    if await self._evaluate_criteria(mock_result, criteria):
                        symbol_signals += 1
                        results["total_signals"] += 1

                        # Check if signal was successful (simplified: positive return next day)
                        if i < len(bars) - 1:
                            next_return = (
                                (bars[i + 1].close - bars[i].close) / bars[i].close
                            ) * 100
                            symbol_returns.append(next_return)

                            if next_return > 0:
                                results["successful_signals"] += 1

                results["symbol_results"][symbol] = {
                    "signals": symbol_signals,
                    "average_return": (
                        sum(symbol_returns) / len(symbol_returns) if symbol_returns else 0
                    ),
                }

            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {e}")

        # Calculate overall statistics
        if results["total_signals"] > 0:
            results["success_rate"] = results["successful_signals"] / results["total_signals"]
            all_returns = [
                r for s in results["symbol_results"].values() for r in s.get("returns", [])
            ]
            if all_returns:
                results["average_return"] = sum(all_returns) / len(all_returns)

        return results
