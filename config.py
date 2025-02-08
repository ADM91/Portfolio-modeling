import os
from typing import List, Dict, Any
from pydantic import BaseSettings


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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Data configuration - moved from hardcoded lists to structured format
action_types: List[Dict[str, str]] = [
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