"""
Domain entities representing core business concepts.
These are the fundamental building blocks of the business domain.
"""

from dataclasses import dataclass
from enum import Enum


class TransactionType(Enum):
    """Types of investment transactions that can be performed."""
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"


@dataclass
class Asset:
    """Domain entity for an asset (stock, currency, etc.)"""
    id: int
    ticker: str
    code: str
    name: str
    is_currency: bool = False
    is_inverted: bool = False

    def __str__(self) -> str:
        return f"{self.ticker} ({self.name})"


@dataclass
class Portfolio:
    """Domain entity for a portfolio"""
    id: int
    name: str
    owner: str

    def __str__(self) -> str:
        return f"{self.name} ({self.owner})"
