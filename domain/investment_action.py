"""
Investment Action domain model with business logic and validation.
This represents the core business concept of an investment transaction.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any
import logging

from .entities import Portfolio, Asset, ActionType


class InvestmentAction:
    """
    Domain model for investment actions (buy, sell, dividend).
    Contains business logic and validation rules.
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        action_type: ActionType,
        date: datetime,
        asset: Asset,
        currency: Asset,
        price: Decimal,
        quantity: Decimal,
        fee: Decimal = Decimal('0'),
        platform: Optional[str] = None,
        comment: Optional[str] = None,
        action_id: Optional[int] = None
    ):
        self.id = action_id
        self.portfolio = portfolio
        self.action_type = action_type
        self.date = date
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
        
        if self.action_type == ActionType.DIVIDEND and self.asset != self.currency:
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
        if self.action_type != ActionType.SELL:
            raise ValueError("Net proceeds only applicable for sell actions")
        return self.calculate_total_value() - self.fee

    def get_quantity_change(self) -> Decimal:
        """Get the quantity change this action represents for portfolio holdings."""
        if self.action_type in (ActionType.BUY, ActionType.DIVIDEND):
            return self.quantity
        elif self.action_type == ActionType.SELL:
            return -self.quantity
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")

    def is_purchase(self) -> bool:
        """Check if this action increases holdings."""
        return self.action_type in (ActionType.BUY, ActionType.DIVIDEND)

    def is_sale(self) -> bool:
        """Check if this action decreases holdings."""
        return self.action_type == ActionType.SELL

    def is_long_term_holding(self, reference_date: Optional[date] = None) -> bool:
        """
        Determine if this would be a long-term holding for tax purposes.
        Uses 1 year (365 days) as the threshold.
        """
        if reference_date is None:
            reference_date = date.today()
        
        action_date = self.date.date() if isinstance(self.date, datetime) else self.date
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
            'action_type': self.action_type.value,
            'date': self.date,
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
    ) -> 'InvestmentAction':
        """Create an InvestmentAction from dictionary data."""
        return cls(
            portfolio=portfolio,
            action_type=ActionType(data['action_type']),
            date=data['date'],
            asset=asset,
            currency=currency,
            price=Decimal(str(data['price'])),
            quantity=Decimal(str(data['quantity'])),
            fee=Decimal(str(data.get('fee', 0))),
            platform=data.get('platform'),
            comment=data.get('comment'),
            action_id=data.get('id')
        )

    def __str__(self) -> str:
        return (f"{self.action_type.value.upper()} {self.quantity} {self.asset.ticker} "
                f"@ {self.price} on {self.date.strftime('%Y-%m-%d')}")

    def __repr__(self) -> str:
        return (f"InvestmentAction(id={self.id}, portfolio={self.portfolio.name}, "
                f"action_type={self.action_type.value}, asset={self.asset.ticker}, "
                f"quantity={self.quantity}, price={self.price}, date={self.date})")
