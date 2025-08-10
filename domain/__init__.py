"""
Domain layer for the Portfolio Modeling application.

This package contains the core business logic and domain models:
- entities.py: Core business entities (Portfolio, Asset, TransactionType)
- investment_transaction.py: Investment transaction domain model with business logic
- portfolio_holding.py: Portfolio holding model with business operations

The domain layer is independent of infrastructure concerns and contains
all the business rules and validation logic.
"""

# Core entities
from .entities import Portfolio, Asset, TransactionType

# Domain models with business logic
from .investment_transaction import InvestmentTransaction
from .portfolio_holding import PortfolioHolding

__all__ = [
    # Core entities
    'Portfolio',
    'Asset', 
    'TransactionType',
    
    # Domain models
    'InvestmentTransaction',
    'PortfolioHolding',
]
