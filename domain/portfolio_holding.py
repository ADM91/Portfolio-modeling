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
from .investment_transaction import InvestmentTransaction


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

    def apply_transaction(self, transaction: InvestmentTransaction) -> 'PortfolioHolding':
        """
        Apply an investment transaction to create a new holding state.
        Returns a new PortfolioHolding with updated quantity.
        """
        if transaction.portfolio.id != self.portfolio.id:
            raise ValueError("Transaction portfolio must match holding portfolio")

        if transaction.asset.id != self.asset.id:
            raise ValueError("Transaction asset must match holding asset")

        new_quantity = self.quantity + transaction.get_quantity_change()

        if new_quantity < 0:
            raise ValueError(f"Transaction would result in negative holdings: {new_quantity}")

        return PortfolioHolding(
            portfolio=self.portfolio,
            asset=self.asset,
            date=transaction.transaction_datetime.date() if isinstance(transaction.transaction_datetime, datetime) else transaction.transaction_datetime,
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
    def validate_transaction_sequence(transactions: List[InvestmentTransaction]) -> List[str]:
        """
        Validate a sequence of actions for business rule violations.
        
        Args:
            transactions: List of transactions to validate (should be sorted by date)
            
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        # Sort transactions by date
        sorted_transactions = sorted(transactions, key=lambda t: t.transaction_datetime)

        # Track holdings by portfolio and asset
        holdings: Dict[Tuple[int, int], Decimal] = {}

        for transaction in sorted_transactions:
            key = (transaction.portfolio.id, transaction.asset.id)
            current_quantity = holdings.get(key, Decimal('0'))

            # Validate individual transaction
            try:
                transaction._validate()
            except ValueError as e:
                errors.append(f"Transaction {transaction} validation failed: {e}")
                continue
            
            # Check for sufficient holdings on sales
            if transaction.is_sale():
                if current_quantity < transaction.quantity:
                    errors.append(
                        f"Insufficient holdings for sale: {transaction}. "
                        f"Current holdings: {current_quantity}, Sale quantity: {transaction.quantity}"
                    )
                    continue
            
            # Update holdings
            holdings[key] = current_quantity + transaction.get_quantity_change()
        
        return errors

    @staticmethod
    def process_transaction_batch(
        transactions: List[InvestmentTransaction],
        current_holdings: Optional[Dict[Tuple[int, int], Decimal]] = None
    ) -> Tuple[List[InvestmentTransaction], List[str]]:
        """
        Process a batch of transactions, validating and marking them as processed.
        
        Args:
            transactions: List of transactions to process
            current_holdings: Current holdings state (optional)
            
        Returns:
            Tuple of (successfully processed transactions, error messages)
        """
        if current_holdings is None:
            current_holdings = {}

        processed_transactions = []
        errors = []

        # Sort transactions by date for proper processing order
        sorted_transactions = sorted(transactions, key=lambda a: a.transaction_datetime)

        for transaction in sorted_transactions:
            # Validate transaction
            if transaction.is_sale():
                key = (transaction.portfolio.id, transaction.asset.id)
                current_quantity = current_holdings.get(key, Decimal('0'))

                if current_quantity < transaction.quantity:
                    error = (f"Insufficient holdings for sale of {transaction.asset.ticker}. "
                           f"Available: {current_quantity}, Requested: {transaction.quantity}")
                    errors.append(error)
                    continue

            # Process transaction
            try:
                # Update holdings
                key = (transaction.portfolio.id, transaction.asset.id)
                current_quantity = current_holdings.get(key, Decimal('0'))
                new_quantity = current_quantity + transaction.get_quantity_change()
                current_holdings[key] = new_quantity
                
                # Mark as processed
                transaction.mark_as_processed()
                processed_transactions.append(transaction)

                logging.info(f"Processed transaction: {transaction}")

            except Exception as e:
                error_msg = f"Error processing transaction {transaction}: {e}"
                errors.append(error_msg)
                logging.error(error_msg)

        return processed_transactions, errors

    @staticmethod
    def calculate_portfolio_summary(transactions: List[InvestmentTransaction]) -> Dict[str, Dict[str, Decimal]]:
        """
        Calculate a summary of portfolio positions from transactions.
        
        Args:
            transactions: List of all transactions for the portfolio

        Returns:
            Dictionary with portfolio summaries by asset
        """
        summary = {}
        
        # Group transactions by portfolio and asset
        by_portfolio_asset = {}
        for transaction in transactions:
            key = (transaction.portfolio.name, transaction.asset.ticker)
            if key not in by_portfolio_asset:
                by_portfolio_asset[key] = []
            by_portfolio_asset[key].append(transaction)

        # Calculate summary for each portfolio-asset combination
        for (portfolio_name, asset_ticker), asset_transactions in by_portfolio_asset.items():
            if portfolio_name not in summary:
                summary[portfolio_name] = {}
            
            total_quantity = Decimal('0')
            total_cost = Decimal('0')
            total_proceeds = Decimal('0')

            for transaction in asset_transactions:
                if transaction.is_purchase():
                    total_quantity += transaction.quantity
                    total_cost += transaction.calculate_total_cost()
                elif transaction.is_sale():
                    total_quantity -= transaction.quantity
                    total_proceeds += transaction.calculate_net_proceeds()

            summary[portfolio_name][asset_ticker] = {
                'quantity': total_quantity,
                'total_cost': total_cost,
                'total_proceeds': total_proceeds,
                'average_cost': total_cost / total_quantity if total_quantity > 0 else Decimal('0')
            }
        
        return summary
