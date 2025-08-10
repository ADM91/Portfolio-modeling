"""
Investment Action domain model with business logic and validation.
This represents the core business concept of an investment transaction.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any
import logging

from .entities import Portfolio, Asset, TransactionType


class InvestmentTransaction:
    """
    Domain model for investment transactions (buy, sell, dividend).
    Contains business logic and validation rules.
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        transaction_type: TransactionType,
        transaction_datetime: datetime,
        asset: Asset,
        currency: Asset,
        price: Decimal,
        quantity: Decimal,
        fee: Decimal = Decimal('0'),
        platform: Optional[str] = None,
        comment: Optional[str] = None,
        transaction_id: Optional[int] = None
    ):
        self.id = transaction_id
        self.portfolio = portfolio
        self.transaction_type = transaction_type
        self.transaction_datetime = transaction_datetime
        self.asset = asset
        self.currency = currency
        self.price = price
        self.quantity = quantity
        self.fee = fee
        self.platform = platform
        self.comment = comment
        self.is_processed = False
        
        self._validate()

    def _validate(self) -> None:
        """Validate business rules for the investment action."""
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")
        
        if self.price < 0:
            raise ValueError(f"Price cannot be negative, got {self.price}")
        
        if self.fee < 0:
            raise ValueError(f"Fee cannot be negative, got {self.fee}")

        if self.transaction_type == TransactionType.DIVIDEND and self.asset != self.currency:
            # For dividends, typically the asset and currency should be the same
            # or we need special handling
            pass  # Allow for now, but could add more specific validation

    def calculate_total_value(self) -> Decimal:
        """Calculate the total value of the action (price * quantity)."""
        return self.price * self.quantity

    def calculate_total_cost(self) -> Decimal:
        """Calculate the total cost including fees."""
        return self.calculate_total_value() + self.fee

    def calculate_net_proceeds(self) -> Decimal:
        """Calculate net proceeds for sales (total value - fees)."""
        if self.transaction_type != TransactionType.SELL:
            raise ValueError("Net proceeds only applicable for sell transactions")
        return self.calculate_total_value() - self.fee

    def get_quantity_change(self) -> Decimal:
        """Get the quantity change this transaction represents for portfolio holdings."""
        if self.transaction_type in (TransactionType.BUY, TransactionType.DIVIDEND):
            return self.quantity
        elif self.transaction_type == TransactionType.SELL:
            return -self.quantity
        else:
            raise ValueError(f"Unknown transaction type: {self.transaction_type}")

    def is_purchase(self) -> bool:
        """Check if this transaction increases holdings."""
        return self.transaction_type in (TransactionType.BUY, TransactionType.DIVIDEND)

    def is_sale(self) -> bool:
        """Check if this transaction decreases holdings."""
        return self.transaction_type == TransactionType.SELL

    def is_long_term_holding(self, reference_date: Optional[date] = None) -> bool:
        """
        Determine if this would be a long-term holding for tax purposes.
        Uses 1 year (365 days) as the threshold.
        """
        if reference_date is None:
            reference_date = date.today()
        
        action_date = self.transaction_datetime.date() if isinstance(self.transaction_datetime, datetime) else self.transaction_datetime
        holding_period = reference_date - action_date
        return holding_period.days > 365

    def mark_as_processed(self) -> None:
        """Mark this action as processed."""
        self.is_processed = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'portfolio_id': self.portfolio.id,
            'transaction_type': self.transaction_type.value,
            'transaction_datetime': self.transaction_datetime,
            'asset_id': self.asset.id,
            'currency_id': self.currency.id,
            'price': self.price,
            'quantity': self.quantity,
            'fee': self.fee,
            'platform': self.platform,
            'comment': self.comment,
            'is_processed': self.is_processed
        }

    @classmethod
    def from_dict(
        cls, 
        data: Dict[str, Any], 
        portfolio: Portfolio, 
        asset: Asset, 
        currency: Asset
    ) -> 'InvestmentTransaction':
        """Create an InvestmentTransaction from dictionary data."""
        return cls(
            portfolio=portfolio,
            transaction_type=TransactionType(data['transaction_type']),
            transaction_datetime=data['transaction_datetime'],
            asset=asset,
            currency=currency,
            price=Decimal(str(data['price'])),
            quantity=Decimal(str(data['quantity'])),
            fee=Decimal(str(data.get('fee', 0))),
            platform=data.get('platform'),
            comment=data.get('comment'),
            transaction_id=data.get('id')
        )

    def __str__(self) -> str:
        return (f"{self.transaction_type.value.upper()} {self.quantity} {self.asset.ticker} "
                f"@ {self.price} on {self.transaction_datetime.strftime('%Y-%m-%d')}")

    def __repr__(self) -> str:
        return (f"InvestmentTransaction(id={self.id}, portfolio={self.portfolio.name}, "
                f"transaction_type={self.transaction_type.value}, asset={self.asset.ticker}, "
                f"quantity={self.quantity}, price={self.price}, date={self.transaction_datetime})")
