"""
Cache management with Redis and in-memory fallback
"""

import json
import logging
import pickle
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Optional

import redis
from redis.exceptions import RedisError

from ..config.constants import CACHE_DEFAULT_TTL, CACHE_MAX_SIZE
from ..config.settings import get_config

logger = logging.getLogger(__name__)


class InMemoryCache:
    """Simple in-memory LRU cache for when Redis is unavailable"""

    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: int = CACHE_DEFAULT_TTL):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._expiry = {}

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self._cache:
            # Check expiry
            if datetime.now() > self._expiry[key]:
                del self._cache[key]
                del self._expiry[key]
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set item in cache"""
        if ttl is None:
            ttl = self.ttl

        # Remove oldest if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            del self._expiry[oldest]

        self._cache[key] = value
        self._expiry[key] = datetime.now() + timedelta(seconds=ttl)
        self._cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        if key in self._cache:
            del self._cache[key]
            del self._expiry[key]
            return True
        return False

    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        self._expiry.clear()

    def cleanup(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [k for k, exp in self._expiry.items() if now > exp]
        for key in expired_keys:
            del self._cache[key]
            del self._expiry[key]


class CacheManager:
    """
    Hybrid cache manager using Redis with in-memory fallback.
    Handles serialization/deserialization automatically.
    """

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize cache manager.

        Args:
            redis_url: Redis connection URL (redis://localhost:6379)
        """
        self.config = get_config()
        self.redis_url = redis_url or self.config.cache.redis_url
        self.redis_client = None
        self.in_memory_cache = InMemoryCache(
            max_size=self.config.cache.max_size, ttl=self.config.cache.ttl
        )

        # Try to connect to Redis
        if self.redis_url and self.config.cache.enabled:
            self._connect_redis()

    def _connect_redis(self):
        """Connect to Redis server"""
        try:
            # Parse Redis URL and create connection pool
            pool = redis.ConnectionPool.from_url(
                self.redis_url, max_connections=50, decode_responses=False  # We'll handle encoding
            )
            self.redis_client = redis.Redis(connection_pool=pool)

            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis cache")

        except (RedisError, Exception) as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache only.")
            self.redis_client = None

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            # Try JSON first (for simple types)
            if isinstance(value, (str, int, float, bool, list, dict)):
                return json.dumps(value).encode("utf-8")
            else:
                # Fall back to pickle for complex objects
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Failed to serialize value: {e}")
            raise

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        if not data:
            return None

        try:
            # Try JSON first
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, AttributeError):
            # Fall back to pickle
            try:
                return pickle.loads(data)
            except Exception as e:
                logger.error(f"Failed to deserialize value: {e}")
                return None

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        # Try Redis first
        if self.redis_client:
            try:
                data = self.redis_client.get(f"scanner:{key}")
                if data:
                    return self._deserialize_value(data)
            except RedisError as e:
                logger.warning(f"Redis get error: {e}")

        # Fall back to in-memory cache
        return self.in_memory_cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for default)
        """
        if ttl is None:
            ttl = self.config.cache.ttl

        # Serialize value
        try:
            serialized = self._serialize_value(value)
        except Exception as e:
            logger.error(f"Cannot cache value for key {key} - serialization failed: {e}")
            return

        # Try Redis first
        if self.redis_client:
            try:
                self.redis_client.setex(f"scanner:{key}", ttl, serialized)
                return
            except RedisError as e:
                logger.warning(f"Redis set error: {e}")

        # Fall back to in-memory cache
        self.in_memory_cache.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        deleted = False

        # Try Redis first
        if self.redis_client:
            try:
                deleted = bool(self.redis_client.delete(f"scanner:{key}"))
            except RedisError as e:
                logger.warning(f"Redis delete error: {e}")

        # Also delete from in-memory cache
        deleted = self.in_memory_cache.delete(key) or deleted

        return deleted

    async def clear_pattern(self, pattern: str):
        """
        Clear all keys matching a pattern.

        Args:
            pattern: Key pattern (e.g., "ticker:*")
        """
        # Clear from Redis
        if self.redis_client:
            try:
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(
                        cursor, match=f"scanner:{pattern}", count=100
                    )
                    if keys:
                        self.redis_client.delete(*keys)
                    if cursor == 0:
                        break
            except RedisError as e:
                logger.warning(f"Redis clear pattern error: {e}")

        # Clear from in-memory cache (simple implementation)
        keys_to_delete = [
            k for k in self.in_memory_cache._cache.keys() if self._match_pattern(k, pattern)
        ]
        for key in keys_to_delete:
            self.in_memory_cache.delete(key)

    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching for in-memory cache"""
        import fnmatch

        return fnmatch.fnmatch(key, pattern)

    async def get_scan_result(self, symbol: str):
        """Get cached scan result for a symbol"""
        return await self.get(f"scan_result:{symbol}")

    async def set_scan_result(self, symbol: str, result: Any, ttl: int = 30):
        """Cache scan result for a symbol"""
        await self.set(f"scan_result:{symbol}", result, ttl)

    async def get_ticker_data(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get cached ticker data"""
        return await self.get(f"ticker:{symbol}")

    async def set_ticker_data(self, symbol: str, data: dict[str, Any], ttl: int = 5):
        """Cache ticker data"""
        await self.set(f"ticker:{symbol}", data, ttl)

    async def get_sentiment(self, symbol: str) -> Optional[float]:
        """Get cached sentiment score"""
        return await self.get(f"sentiment:{symbol}")

    async def set_sentiment(self, symbol: str, score: float, ttl: int = 900):
        """Cache sentiment score (15 minutes default)"""
        await self.set(f"sentiment:{symbol}", score, ttl)

    async def get_ml_prediction(self, symbol: str, model: str) -> Optional[dict[str, float]]:
        """Get cached ML prediction"""
        return await self.get(f"ml:{model}:{symbol}")

    async def set_ml_prediction(
        self, symbol: str, model: str, prediction: dict[str, float], ttl: int = 300
    ):
        """Cache ML prediction (5 minutes default)"""
        await self.set(f"ml:{model}:{symbol}", prediction, ttl)

    async def increment_counter(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter in cache.

        Args:
            key: Counter key
            amount: Amount to increment by

        Returns:
            New counter value
        """
        if self.redis_client:
            try:
                return self.redis_client.incrby(f"scanner:counter:{key}", amount)
            except RedisError as e:
                logger.warning(f"Redis increment error: {e}")

        # Fallback for in-memory
        current = self.in_memory_cache.get(f"counter:{key}") or 0
        new_value = current + amount
        self.in_memory_cache.set(f"counter:{key}", new_value)
        return new_value

    async def get_list(self, key: str) -> list[Any]:
        """Get list from cache"""
        value = await self.get(key)
        return value if isinstance(value, list) else []

    async def append_to_list(self, key: str, item: Any, max_size: int = 100):
        """Append item to cached list with size limit"""
        current_list = await self.get_list(key)
        current_list.append(item)

        # Trim to max size
        if len(current_list) > max_size:
            current_list = current_list[-max_size:]

        await self.set(key, current_list)

    def cleanup(self):
        """Clean up expired entries"""
        self.in_memory_cache.cleanup()

        # Redis handles expiry automatically

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "redis_connected": self.redis_client is not None,
            "in_memory_items": len(self.in_memory_cache._cache),
            "in_memory_size": self.in_memory_cache.max_size,
        }

        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update(
                    {
                        "redis_used_memory": info.get("used_memory_human"),
                        "redis_connected_clients": info.get("connected_clients"),
                        "redis_total_commands": info.get("total_commands_processed"),
                    }
                )
            except Exception:
                pass

        return stats
