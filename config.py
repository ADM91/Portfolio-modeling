import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Database configuration
    database_url: str = "sqlite:///./portfolio.db"

    # API configuration
    api_key_yfinance: str = ""
    debug: bool = False

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "pretty"  # 'pretty' for development, 'json' for production
    log_file: str = "logs/portfolio_tracker.log"
    log_max_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    log_sql_queries: bool = False  # Enable SQL query logging
    log_performance: bool = True   # Enable performance timing logs
    log_api_requests: bool = True  # Enable API request/response logging
    
    # Component-specific log levels
    log_level_database: str = "INFO"
    log_level_services: str = "INFO"
    log_level_api: str = "INFO"
    log_level_external: str = "INFO"  # For external API calls

    # Path configuration
    path_actions: str = "_data/test_action_data.xlsx"
    pythonpath: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # This allows extra environment variables
    }
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug or self.log_format == "pretty"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug and self.log_format == "json"


# Global settings instance
settings = Settings()

# Data configuration - moved from hardcoded lists to structured format
transaction_types: List[Dict[str, str]] = [
    {"name": "buy"},
    {"name": "sell"},
    {"name": "dividend"}
]

assets: List[Dict[str, Any]] = [
    {"ticker": "", "code": "USD", "name": "US dollar", "is_currency": True, "is_inverted": False},  # special case, value = 1
    {"ticker": "ISK=X", "code": "ISK", "name": "Icelandic krona", "is_currency": True, "is_inverted": True},
    {"ticker": "EUR=X", "code": "EUR", "name": "Euro", "is_currency": True, "is_inverted": True},
    {"ticker": "BTC-USD", "code": "BTC", "name": "Bitcoin", "is_currency": True, "is_inverted": False},
    {"ticker": "ETH-USD", "code": "ETH", "name": "Ethereum", "is_currency": False, "is_inverted": False},
    {"ticker": "MATIC-USD", "code": "MATIC", "name": "Polygon", "is_currency": False, "is_inverted": False},
    {"ticker": "SPY", "code": "SPY", "name": "SP-500", "is_currency": False, "is_inverted": False},
]

portfolios: List[Dict[str, str]] = [
    {"name": "Alexander", "owner": "Alexander"},
    {"name": "AM ehf.", "owner": "AM ehf."},
    {"name": "Pauline", "owner": "Pauline"},
    {"name": "Óliver", "owner": "Óliver"},
]
