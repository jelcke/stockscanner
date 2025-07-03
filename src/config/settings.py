"""
Configuration management - loads settings from environment and config files
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .constants import (
    ALERT_DEDUP_WINDOW_MINUTES,
    CACHE_DEFAULT_TTL,
    CACHE_MAX_SIZE,
    DB_DEFAULT_PATH,
    DB_MAX_OVERFLOW,
    DB_POOL_SIZE,
    DEFAULT_MAX_RESULTS,
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_UNIVERSE,
    ENV_IB_HOST,
    ENV_IB_PORT,
    ENV_REDIS_URL,
    ENV_TELEGRAM_BOT_TOKEN,
    IB_DEFAULT_CLIENT_ID,
    IB_DEFAULT_HOST,
    IB_DEFAULT_PORT_PAPER,
    IB_MAX_RECONNECT_ATTEMPTS,
    LOG_DEFAULT_LEVEL,
    LOG_FORMAT,
    MAX_MARKET_DATA_LINES_STANDARD,
)

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class IBConnectionConfig(BaseModel):
    """IB connection configuration"""

    host: str = Field(default=IB_DEFAULT_HOST)
    port: int = Field(default=IB_DEFAULT_PORT_PAPER)
    client_id: int = Field(default=IB_DEFAULT_CLIENT_ID)
    timeout: int = Field(default=30)
    max_reconnect_attempts: int = Field(default=IB_MAX_RECONNECT_ATTEMPTS)
    use_delayed_data: bool = Field(default=True)


class ScannerConfig(BaseModel):
    """Scanner configuration"""

    default_universe: str = Field(default=DEFAULT_UNIVERSE)
    max_results: int = Field(default=DEFAULT_MAX_RESULTS)
    scan_interval: int = Field(default=DEFAULT_SCAN_INTERVAL)
    data_update_interval: int = Field(default=5)
    max_market_data_lines: int = Field(default=MAX_MARKET_DATA_LINES_STANDARD)


class DatabaseConfig(BaseModel):
    """Database configuration"""

    type: str = Field(default="sqlite")
    path: str = Field(default=DB_DEFAULT_PATH)
    pool_size: int = Field(default=DB_POOL_SIZE)
    max_overflow: int = Field(default=DB_MAX_OVERFLOW)


class CacheConfig(BaseModel):
    """Cache configuration"""

    enabled: bool = Field(default=True)
    ttl: int = Field(default=CACHE_DEFAULT_TTL)
    max_size: int = Field(default=CACHE_MAX_SIZE)
    redis_url: Optional[str] = Field(default=None)


class AlertConfig(BaseModel):
    """Alert configuration"""

    enabled: bool = Field(default=True)
    dedup_minutes: int = Field(default=ALERT_DEDUP_WINDOW_MINUTES)
    email_enabled: bool = Field(default=False)
    telegram_enabled: bool = Field(default=False)
    webhook_enabled: bool = Field(default=False)


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: str = Field(default=LOG_DEFAULT_LEVEL)
    format: str = Field(default=LOG_FORMAT)


class AppConfig(BaseModel):
    """Main application configuration"""

    ib_connection: IBConnectionConfig = Field(default_factory=IBConnectionConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class Settings:
    """Settings manager that merges environment variables and config files"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self._config: Optional[AppConfig] = None
        self._load_config()

    def _load_config(self):
        """Load configuration from file and environment"""
        config_dict = {}

        # Load from YAML file if exists
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config_dict = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")

        # Override with environment variables
        self._merge_env_vars(config_dict)

        # Create pydantic model
        self._config = AppConfig(**config_dict)

    def _merge_env_vars(self, config_dict: dict[str, Any]):
        """Merge environment variables into config dictionary"""
        # IB Connection settings
        if not config_dict.get("ib_connection"):
            config_dict["ib_connection"] = {}

        if os.getenv(ENV_IB_HOST):
            config_dict["ib_connection"]["host"] = os.getenv(ENV_IB_HOST)
        if os.getenv(ENV_IB_PORT):
            config_dict["ib_connection"]["port"] = int(os.getenv(ENV_IB_PORT))

        # Cache settings
        if not config_dict.get("cache"):
            config_dict["cache"] = {}

        if os.getenv(ENV_REDIS_URL):
            config_dict["cache"]["redis_url"] = os.getenv(ENV_REDIS_URL)

        # Alert settings
        if not config_dict.get("alerts"):
            config_dict["alerts"] = {}

        if os.getenv(ENV_TELEGRAM_BOT_TOKEN):
            config_dict["alerts"]["telegram_enabled"] = True

    @property
    def config(self) -> AppConfig:
        """Get the loaded configuration"""
        if not self._config:
            self._load_config()
        return self._config

    def get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key from environment"""
        return os.getenv(key_name)

    def get_ib_connection_params(self) -> dict:
        """Get IB connection parameters"""
        return {
            "host": self.config.ib_connection.host,
            "port": self.config.ib_connection.port,
            "clientId": self.config.ib_connection.client_id,
            "timeout": self.config.ib_connection.timeout,
        }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level), format=self.config.logging.format
        )


# Global settings instance
settings = Settings()


# Convenience functions
def get_config() -> AppConfig:
    """Get application configuration"""
    return settings.config


def get_api_key(key_name: str) -> Optional[str]:
    """Get API key from environment"""
    return settings.get_api_key(key_name)
