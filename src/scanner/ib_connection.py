"""
Interactive Brokers connection management with rate limiting and reconnection
"""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Optional

from ib_async import IB, Contract, Stock, Ticker

from ..config.constants import (
    IB_RATE_LIMIT_PER_SECOND,
    IB_RATE_LIMIT_WINDOW,
    IB_RECONNECT_MAX_WAIT,
    MARKET_DATA_TYPE_DELAYED,
)
from ..config.settings import get_config

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for IB API requests"""

    def __init__(
        self, max_requests: int = IB_RATE_LIMIT_PER_SECOND, window: float = IB_RATE_LIMIT_WINDOW
    ):
        self.max_requests = max_requests
        self.window = window
        self.requests = deque(maxlen=max_requests)
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        async with self._lock:
            now = asyncio.get_event_loop().time()

            # Remove old requests outside the window
            while self.requests and self.requests[0] <= now - self.window:
                self.requests.popleft()

            # If at limit, wait until we can make another request
            if len(self.requests) >= self.max_requests:
                sleep_time = self.window - (now - self.requests[0]) + 0.01
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping for {sleep_time:.3f}s")
                    await asyncio.sleep(sleep_time)
                    # Recursive call to re-check after sleep
                    await self.acquire()
                    return

            # Record this request
            self.requests.append(now)


class MarketDataManager:
    """Manage market data line subscriptions"""

    def __init__(self, max_lines: int):
        self.max_lines = max_lines
        self.active_subscriptions: dict[str, Ticker] = {}
        self.last_access: dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def can_subscribe(self) -> bool:
        """Check if we can subscribe to another symbol"""
        async with self._lock:
            return len(self.active_subscriptions) < self.max_lines

    async def get_lru_symbol(self) -> Optional[str]:
        """Get least recently used symbol"""
        async with self._lock:
            if not self.last_access:
                return None
            return min(self.last_access.items(), key=lambda x: x[1])[0]

    async def add_subscription(self, symbol: str, ticker: Ticker):
        """Add a subscription"""
        async with self._lock:
            self.active_subscriptions[symbol] = ticker
            self.last_access[symbol] = datetime.now()

    async def remove_subscription(self, symbol: str) -> Optional[Ticker]:
        """Remove a subscription"""
        async with self._lock:
            ticker = self.active_subscriptions.pop(symbol, None)
            self.last_access.pop(symbol, None)
            return ticker

    async def touch(self, symbol: str):
        """Update last access time for a symbol"""
        async with self._lock:
            if symbol in self.active_subscriptions:
                self.last_access[symbol] = datetime.now()

    def get_active_count(self) -> int:
        """Get number of active subscriptions"""
        return len(self.active_subscriptions)


class IBConnectionManager:
    """Manages IB API connection with automatic reconnection and rate limiting"""

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize connection manager.

        Args:
            config: Configuration dictionary, if None uses default from settings
        """
        self.config = config or get_config().ib_connection.dict()
        self.ib = IB()
        self.rate_limiter = RateLimiter()
        self.market_data_manager = MarketDataManager(self.config.get("max_market_data_lines", 100))

        self._connected = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Event handlers
        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error

    async def connect(self) -> bool:
        """
        Connect to IB Gateway/TWS with exponential backoff retry.

        Returns:
            True if connected successfully, False otherwise
        """
        max_attempts = self.config.get("max_reconnect_attempts", 5)

        for attempt in range(max_attempts):
            try:
                logger.info(
                    f"Connecting to IB Gateway at {self.config['host']}:{self.config['port']} "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )

                await self.ib.connectAsync(
                    self.config["host"],
                    self.config["port"],
                    clientId=self.config["client_id"],
                    timeout=self.config.get("timeout", 30),
                )

                # Use delayed data if configured
                if self.config.get("use_delayed_data", True):
                    self.ib.reqMarketDataType(MARKET_DATA_TYPE_DELAYED)
                    logger.info("Using delayed market data (15-20 minute delay)")

                self._connected = True
                logger.info("Successfully connected to IB Gateway")
                return True

            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")

                if attempt < max_attempts - 1:
                    # Exponential backoff with jitter
                    wait_time = min(2**attempt + (attempt * 0.1), IB_RECONNECT_MAX_WAIT)
                    logger.info(f"Waiting {wait_time:.1f} seconds before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max connection attempts reached")

        return False

    async def disconnect(self):
        """Disconnect from IB Gateway"""
        self._shutdown = True

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()

        if self.ib.isConnected():
            # Unsubscribe all market data
            for _symbol, ticker in list(self.market_data_manager.active_subscriptions.items()):
                self.ib.cancelMktData(ticker.contract)

            self.ib.disconnect()
            logger.info("Disconnected from IB Gateway")

    async def ensure_connected(self) -> bool:
        """Ensure connection is active, reconnect if necessary"""
        if self.ib.isConnected():
            return True

        logger.warning("Not connected, attempting to reconnect...")
        return await self.connect()

    async def subscribe_market_data(
        self,
        symbol: str,
        generic_tick_list: str = "",
        snapshot: bool = False,
        regulatory_snapshot: bool = False,
    ) -> Optional[Ticker]:
        """
        Subscribe to market data for a symbol.

        Args:
            symbol: Stock symbol
            generic_tick_list: Generic tick types to request
            snapshot: Request snapshot instead of streaming data
            regulatory_snapshot: Request regulatory snapshot

        Returns:
            Ticker object if successful, None otherwise
        """
        # Rate limiting
        await self.rate_limiter.acquire()

        # Ensure connected
        if not await self.ensure_connected():
            logger.error("Cannot subscribe - not connected")
            return None

        try:
            # Check if already subscribed
            existing = self.market_data_manager.active_subscriptions.get(symbol)
            if existing:
                await self.market_data_manager.touch(symbol)
                return existing

            # Check market data lines limit
            if not await self.market_data_manager.can_subscribe():
                # Try to free up a line by removing LRU
                lru_symbol = await self.market_data_manager.get_lru_symbol()
                if lru_symbol:
                    logger.info(f"Market data lines full, unsubscribing LRU symbol: {lru_symbol}")
                    await self.unsubscribe_market_data(lru_symbol)
                else:
                    logger.error("Market data lines full and no LRU symbol to remove")
                    return None

            # Qualify contract
            contract = Stock(symbol, "SMART", "USD")
            contracts = await self.ib.qualifyContractsAsync(contract)

            if not contracts:
                logger.error(f"Cannot qualify contract for {symbol}")
                return None

            # Subscribe to market data
            ticker = self.ib.reqMktData(
                contracts[0], generic_tick_list, snapshot, regulatory_snapshot
            )

            # Add to active subscriptions
            await self.market_data_manager.add_subscription(symbol, ticker)

            logger.info(
                f"Subscribed to market data for {symbol} "
                f"({self.market_data_manager.get_active_count()}/{self.market_data_manager.max_lines} lines used)"
            )

            return ticker

        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {e}")
            return None

    async def unsubscribe_market_data(self, symbol: str) -> bool:
        """
        Unsubscribe from market data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            True if unsubscribed successfully
        """
        ticker = await self.market_data_manager.remove_subscription(symbol)

        if ticker:
            try:
                self.ib.cancelMktData(ticker.contract)
                logger.info(
                    f"Unsubscribed from {symbol} "
                    f"({self.market_data_manager.get_active_count()}/{self.market_data_manager.max_lines} lines used)"
                )
                return True
            except Exception as e:
                logger.error(f"Error unsubscribing from {symbol}: {e}")

        return False

    async def get_historical_data(
        self,
        symbol: str,
        end_datetime: str = "",
        duration_str: str = "1 D",
        bar_size_setting: str = "1 min",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        format_date: int = 1,
    ) -> Optional[list]:
        """
        Get historical data for a symbol.

        Args:
            symbol: Stock symbol
            end_datetime: End date/time for historical data
            duration_str: Duration string (e.g., '1 D', '1 W', '1 M')
            bar_size_setting: Bar size (e.g., '1 min', '5 mins', '1 hour', '1 day')
            what_to_show: Type of data (TRADES, MIDPOINT, BID, ASK, etc.)
            use_rth: Use regular trading hours only
            format_date: Date format (1 for yyyyMMdd HH:mm:ss, 2 for Unix time)

        Returns:
            List of BarData objects if successful, None otherwise
        """
        # Rate limiting
        await self.rate_limiter.acquire()

        # Ensure connected
        if not await self.ensure_connected():
            logger.error("Cannot get historical data - not connected")
            return None

        try:
            # Qualify contract
            contract = Stock(symbol, "SMART", "USD")
            contracts = await self.ib.qualifyContractsAsync(contract)

            if not contracts:
                logger.error(f"Cannot qualify contract for {symbol}")
                return None

            # Get historical data
            bars = await self.ib.reqHistoricalDataAsync(
                contracts[0],
                endDateTime=end_datetime,
                durationStr=duration_str,
                barSizeSetting=bar_size_setting,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=format_date,
                keepUpToDate=False,
            )

            logger.info(f"Retrieved {len(bars)} bars of historical data for {symbol}")
            return bars

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    async def get_contract_details(self, symbol: str) -> Optional[list]:
        """
        Get contract details for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            List of ContractDetails objects if successful
        """
        # Rate limiting
        await self.rate_limiter.acquire()

        # Ensure connected
        if not await self.ensure_connected():
            return None

        try:
            contract = Stock(symbol, "SMART", "USD")
            details = await self.ib.reqContractDetailsAsync(contract)
            return details
        except Exception as e:
            logger.error(f"Error getting contract details for {symbol}: {e}")
            return None

    def _on_connected(self):
        """Handle connection event"""
        logger.info("Connected event received")
        self._connected = True

    def _on_disconnected(self):
        """Handle disconnection event"""
        logger.warning("Disconnected event received")
        self._connected = False

        # Start reconnection task if not shutting down
        if not self._shutdown and (not self._reconnect_task or self._reconnect_task.done()):
            self._reconnect_task = asyncio.create_task(self._auto_reconnect())

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract):
        """Handle error events"""
        # Log based on error code severity
        if errorCode < 1000:  # System errors
            logger.error(f"System error {errorCode}: {errorString}")
        elif errorCode < 2000:  # Warning
            logger.warning(f"Warning {errorCode}: {errorString}")
        else:  # Info
            logger.info(f"Info {errorCode}: {errorString}")

    async def _auto_reconnect(self):
        """Auto reconnect with exponential backoff"""
        attempt = 0

        while not self._shutdown and not self._connected:
            wait_time = min(2**attempt, IB_RECONNECT_MAX_WAIT)
            logger.info(f"Auto-reconnect in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

            if await self.connect():
                logger.info("Auto-reconnect successful")
                break
            else:
                attempt += 1

    @property
    def is_connected(self) -> bool:
        """Check if connected to IB Gateway"""
        return self.ib.isConnected()

    def get_active_subscriptions(self) -> set[str]:
        """Get set of actively subscribed symbols"""
        return set(self.market_data_manager.active_subscriptions.keys())
