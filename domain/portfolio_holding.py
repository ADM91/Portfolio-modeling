"""
Portfolio Holding domain model with business operations.
This represents the state of holdings in a portfolio and provides
business logic for processing investment actions.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import List, Dict, Any, Tuple, Optional
import logging

from .entities import Portfolio, Asset
from .investment_action import InvestmentAction


class PortfolioHolding:
    """
    Domain model for portfolio holdings at a specific point in time.
    Contains business logic for processing investment actions.
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        asset: Asset,
        date: date,
        quantity: Decimal
    ):
        self.portfolio = portfolio
        self.asset = asset
        self.date = date
        self.quantity = quantity

    def apply_action(self, action: InvestmentAction) -> 'PortfolioHolding':
        """
        Apply an investment action to create a new holding state.
        Returns a new PortfolioHolding with updated quantity.
        """
        if action.portfolio.id != self.portfolio.id:
            raise ValueError("Action portfolio must match holding portfolio")
        
        if action.asset.id != self.asset.id:
            raise ValueError("Action asset must match holding asset")
        
        new_quantity = self.quantity + action.get_quantity_change()
        
        if new_quantity < 0:
            raise ValueError(f"Action would result in negative holdings: {new_quantity}")
        
        return PortfolioHolding(
            portfolio=self.portfolio,
            asset=self.asset,
            date=action.date.date() if isinstance(action.date, datetime) else action.date,
            quantity=new_quantity
        )

    def is_empty(self) -> bool:
        """Check if this holding has zero quantity."""
        return self.quantity == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'portfolio_id': self.portfolio.id,
            'asset_id': self.asset.id,
            'date': self.date,
            'quantity': self.quantity
        }

    def __str__(self) -> str:
        return f"{self.quantity} {self.asset.ticker} in {self.portfolio.name} on {self.date}"

    def __repr__(self) -> str:
        return (f"PortfolioHolding(portfolio={self.portfolio.name}, "
                f"asset={self.asset.ticker}, date={self.date}, quantity={self.quantity})")

    @staticmethod
    def validate_action_sequence(actions: List[InvestmentAction]) -> List[str]:
        """
        Validate a sequence of actions for business rule violations.
        
        Args:
            actions: List of actions to validate (should be sorted by date)
            
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        
        # Sort actions by date
        sorted_actions = sorted(actions, key=lambda a: a.date)
        
        # Track holdings by portfolio and asset
        holdings: Dict[Tuple[int, int], Decimal] = {}
        
        for action in sorted_actions:
            key = (action.portfolio.id, action.asset.id)
            current_quantity = holdings.get(key, Decimal('0'))
            
            # Validate individual action
            try:
                action._validate()
            except ValueError as e:
                errors.append(f"Action {action} validation failed: {e}")
                continue
            
            # Check for sufficient holdings on sales
            if action.is_sale():
                if current_quantity < action.quantity:
                    errors.append(
                        f"Insufficient holdings for sale: {action}. "
                        f"Current holdings: {current_quantity}, Sale quantity: {action.quantity}"
                    )
                    continue
            
            # Update holdings
            holdings[key] = current_quantity + action.get_quantity_change()
        
        return errors

    @staticmethod
    def process_action_batch(
        actions: List[InvestmentAction],
        current_holdings: Optional[Dict[Tuple[int, int], Decimal]] = None
    ) -> Tuple[List[InvestmentAction], List[str]]:
        """
        Process a batch of actions, validating and marking them as processed.
        
        Args:
            actions: List of actions to process
            current_holdings: Current holdings state (optional)
            
        Returns:
            Tuple of (successfully processed actions, error messages)
        """
        if current_holdings is None:
            current_holdings = {}
        
        processed_actions = []
        errors = []
        
        # Sort actions by date for proper processing order
        sorted_actions = sorted(actions, key=lambda a: a.date)
        
        for action in sorted_actions:
            # Validate action
            if action.is_sale():
                key = (action.portfolio.id, action.asset.id)
                current_quantity = current_holdings.get(key, Decimal('0'))
                
                if current_quantity < action.quantity:
                    error = (f"Insufficient holdings for sale of {action.asset.ticker}. "
                           f"Available: {current_quantity}, Requested: {action.quantity}")
                    errors.append(error)
                    continue
            
            # Process action
            try:
                # Update holdings
                key = (action.portfolio.id, action.asset.id)
                current_quantity = current_holdings.get(key, Decimal('0'))
                new_quantity = current_quantity + action.get_quantity_change()
                current_holdings[key] = new_quantity
                
                # Mark as processed
                action.mark_as_processed()
                processed_actions.append(action)
                
                logging.info(f"Processed action: {action}")
                
            except Exception as e:
                error_msg = f"Error processing action {action}: {e}"
                errors.append(error_msg)
                logging.error(error_msg)
        
        return processed_actions, errors

    @staticmethod
    def calculate_portfolio_summary(actions: List[InvestmentAction]) -> Dict[str, Dict[str, Decimal]]:
        """
        Calculate a summary of portfolio positions from actions.
        
        Args:
            actions: List of all actions for the portfolio
            
        Returns:
            Dictionary with portfolio summaries by asset
        """
        summary = {}
        
        # Group actions by portfolio and asset
        by_portfolio_asset = {}
        for action in actions:
            key = (action.portfolio.name, action.asset.ticker)
            if key not in by_portfolio_asset:
                by_portfolio_asset[key] = []
            by_portfolio_asset[key].append(action)
        
        # Calculate summary for each portfolio-asset combination
        for (portfolio_name, asset_ticker), asset_actions in by_portfolio_asset.items():
            if portfolio_name not in summary:
                summary[portfolio_name] = {}
            
            total_quantity = Decimal('0')
            total_cost = Decimal('0')
            total_proceeds = Decimal('0')
            
            for action in asset_actions:
                if action.is_purchase():
                    total_quantity += action.quantity
                    total_cost += action.calculate_total_cost()
                elif action.is_sale():
                    total_quantity -= action.quantity
                    total_proceeds += action.calculate_net_proceeds()
            
            summary[portfolio_name][asset_ticker] = {
                'quantity': total_quantity,
                'total_cost': total_cost,
                'total_proceeds': total_proceeds,
                'average_cost': total_cost / total_quantity if total_quantity > 0 else Decimal('0')
            }
        
        return summary
