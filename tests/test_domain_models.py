"""
Unit tests for domain models.
These tests can run without database dependencies and focus on business logic.
"""

import pytest
from datetime import datetime, date
from decimal import Decimal

from domain import InvestmentAction, Portfolio, Asset, ActionType, PortfolioHolding


class TestInvestmentAction:
    """Test cases for InvestmentAction domain model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.portfolio = Portfolio(id=1, name="Test Portfolio", owner="Test Owner")
        self.asset = Asset(id=1, ticker="AAPL", code="AAPL", name="Apple Inc.")
        self.currency = Asset(id=2, ticker="USD", code="USD", name="US Dollar", is_currency=True)
    
    def test_create_buy_action(self):
        """Test creating a buy action."""
        action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.BUY,
            date=datetime(2024, 1, 15),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('150.00'),
            quantity=Decimal('10'),
            fee=Decimal('5.00')
        )
        
        assert action.portfolio == self.portfolio
        assert action.action_type == ActionType.BUY
        assert action.asset == self.asset
        assert action.price == Decimal('150.00')
        assert action.quantity == Decimal('10')
        assert action.fee == Decimal('5.00')
        assert not action.is_processed
    
    def test_create_sell_action(self):
        """Test creating a sell action."""
        action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.SELL,
            date=datetime(2024, 2, 15),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('160.00'),
            quantity=Decimal('5'),
            fee=Decimal('3.00')
        )
        
        assert action.action_type == ActionType.SELL
        assert action.is_sale()
        assert not action.is_purchase()
    
    def test_validation_negative_quantity(self):
        """Test validation fails for negative quantity."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            InvestmentAction(
                portfolio=self.portfolio,
                action_type=ActionType.BUY,
                date=datetime(2024, 1, 15),
                asset=self.asset,
                currency=self.currency,
                price=Decimal('150.00'),
                quantity=Decimal('-10'),  # Invalid
                fee=Decimal('5.00')
            )
    
    def test_validation_negative_price(self):
        """Test validation fails for negative price."""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            InvestmentAction(
                portfolio=self.portfolio,
                action_type=ActionType.BUY,
                date=datetime(2024, 1, 15),
                asset=self.asset,
                currency=self.currency,
                price=Decimal('-150.00'),  # Invalid
                quantity=Decimal('10'),
                fee=Decimal('5.00')
            )
    
    def test_validation_negative_fee(self):
        """Test validation fails for negative fee."""
        with pytest.raises(ValueError, match="Fee cannot be negative"):
            InvestmentAction(
                portfolio=self.portfolio,
                action_type=ActionType.BUY,
                date=datetime(2024, 1, 15),
                asset=self.asset,
                currency=self.currency,
                price=Decimal('150.00'),
                quantity=Decimal('10'),
                fee=Decimal('-5.00')  # Invalid
            )
    
    def test_calculate_total_value(self):
        """Test total value calculation."""
        action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.BUY,
            date=datetime(2024, 1, 15),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('150.00'),
            quantity=Decimal('10'),
            fee=Decimal('5.00')
        )
        
        assert action.calculate_total_value() == Decimal('1500.00')
    
    def test_calculate_total_cost(self):
        """Test total cost calculation including fees."""
        action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.BUY,
            date=datetime(2024, 1, 15),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('150.00'),
            quantity=Decimal('10'),
            fee=Decimal('5.00')
        )
        
        assert action.calculate_total_cost() == Decimal('1505.00')
    
    def test_calculate_net_proceeds(self):
        """Test net proceeds calculation for sales."""
        action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.SELL,
            date=datetime(2024, 2, 15),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('160.00'),
            quantity=Decimal('5'),
            fee=Decimal('3.00')
        )
        
        assert action.calculate_net_proceeds() == Decimal('797.00')  # 800 - 3
    
    def test_net_proceeds_invalid_for_buy(self):
        """Test that net proceeds calculation fails for buy actions."""
        action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.BUY,
            date=datetime(2024, 1, 15),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('150.00'),
            quantity=Decimal('10'),
            fee=Decimal('5.00')
        )
        
        with pytest.raises(ValueError, match="Net proceeds only applicable for sell actions"):
            action.calculate_net_proceeds()
    
    def test_get_quantity_change_buy(self):
        """Test quantity change for buy action."""
        action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.BUY,
            date=datetime(2024, 1, 15),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('150.00'),
            quantity=Decimal('10'),
            fee=Decimal('5.00')
        )
        
        assert action.get_quantity_change() == Decimal('10')
    
    def test_get_quantity_change_sell(self):
        """Test quantity change for sell action."""
        action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.SELL,
            date=datetime(2024, 2, 15),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('160.00'),
            quantity=Decimal('5'),
            fee=Decimal('3.00')
        )
        
        assert action.get_quantity_change() == Decimal('-5')
    
    def test_is_long_term_holding(self):
        """Test long-term holding determination."""
        # Action from over a year ago
        old_action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.BUY,
            date=datetime(2022, 1, 15),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('150.00'),
            quantity=Decimal('10'),
            fee=Decimal('5.00')
        )
        
        # Action from less than a year ago
        recent_action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.BUY,
            date=datetime(2024, 6, 15),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('150.00'),
            quantity=Decimal('10'),
            fee=Decimal('5.00')
        )
        
        reference_date = date(2024, 12, 31)
        assert old_action.is_long_term_holding(reference_date)
        assert not recent_action.is_long_term_holding(reference_date)
    
    def test_mark_as_processed(self):
        """Test marking action as processed."""
        action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.BUY,
            date=datetime(2024, 1, 15),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('150.00'),
            quantity=Decimal('10'),
            fee=Decimal('5.00')
        )
        
        assert not action.is_processed
        action.mark_as_processed()
        assert action.is_processed
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.BUY,
            date=datetime(2024, 1, 15),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('150.00'),
            quantity=Decimal('10'),
            fee=Decimal('5.00'),
            platform="Test Platform",
            comment="Test Comment",
            action_id=123
        )
        
        result = action.to_dict()
        
        assert result['id'] == 123
        assert result['portfolio_id'] == 1
        assert result['action_type'] == 'buy'
        assert result['asset_id'] == 1
        assert result['currency_id'] == 2
        assert result['price'] == Decimal('150.00')
        assert result['quantity'] == Decimal('10')
        assert result['fee'] == Decimal('5.00')
        assert result['platform'] == "Test Platform"
        assert result['comment'] == "Test Comment"


class TestPortfolioHolding:
    """Test cases for PortfolioHolding domain model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.portfolio = Portfolio(id=1, name="Test Portfolio", owner="Test Owner")
        self.asset = Asset(id=1, ticker="AAPL", code="AAPL", name="Apple Inc.")
        self.currency = Asset(id=2, ticker="USD", code="USD", name="US Dollar", is_currency=True)
    
    def test_create_holding(self):
        """Test creating a portfolio holding."""
        holding = PortfolioHolding(
            portfolio=self.portfolio,
            asset=self.asset,
            date=date(2024, 1, 15),
            quantity=Decimal('100')
        )
        
        assert holding.portfolio == self.portfolio
        assert holding.asset == self.asset
        assert holding.date == date(2024, 1, 15)
        assert holding.quantity == Decimal('100')
    
    def test_apply_buy_action(self):
        """Test applying a buy action to holdings."""
        holding = PortfolioHolding(
            portfolio=self.portfolio,
            asset=self.asset,
            date=date(2024, 1, 15),
            quantity=Decimal('100')
        )
        
        buy_action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.BUY,
            date=datetime(2024, 1, 16),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('150.00'),
            quantity=Decimal('50'),
            fee=Decimal('5.00')
        )
        
        new_holding = holding.apply_action(buy_action)
        
        assert new_holding.quantity == Decimal('150')  # 100 + 50
        assert new_holding.date == date(2024, 1, 16)
    
    def test_apply_sell_action(self):
        """Test applying a sell action to holdings."""
        holding = PortfolioHolding(
            portfolio=self.portfolio,
            asset=self.asset,
            date=date(2024, 1, 15),
            quantity=Decimal('100')
        )
        
        sell_action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.SELL,
            date=datetime(2024, 1, 16),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('160.00'),
            quantity=Decimal('30'),
            fee=Decimal('3.00')
        )
        
        new_holding = holding.apply_action(sell_action)
        
        assert new_holding.quantity == Decimal('70')  # 100 - 30
        assert new_holding.date == date(2024, 1, 16)
    
    def test_apply_action_insufficient_holdings(self):
        """Test that selling more than available raises error."""
        holding = PortfolioHolding(
            portfolio=self.portfolio,
            asset=self.asset,
            date=date(2024, 1, 15),
            quantity=Decimal('50')
        )
        
        sell_action = InvestmentAction(
            portfolio=self.portfolio,
            action_type=ActionType.SELL,
            date=datetime(2024, 1, 16),
            asset=self.asset,
            currency=self.currency,
            price=Decimal('160.00'),
            quantity=Decimal('100'),  # More than available
            fee=Decimal('3.00')
        )
        
        with pytest.raises(ValueError, match="Action would result in negative holdings"):
            holding.apply_action(sell_action)
    
    def test_is_empty(self):
        """Test checking if holding is empty."""
        empty_holding = PortfolioHolding(
            portfolio=self.portfolio,
            asset=self.asset,
            date=date(2024, 1, 15),
            quantity=Decimal('0')
        )
        
        non_empty_holding = PortfolioHolding(
            portfolio=self.portfolio,
            asset=self.asset,
            date=date(2024, 1, 15),
            quantity=Decimal('100')
        )
        
        assert empty_holding.is_empty()
        assert not non_empty_holding.is_empty()
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        holding = PortfolioHolding(
            portfolio=self.portfolio,
            asset=self.asset,
            date=date(2024, 1, 15),
            quantity=Decimal('100')
        )
        
        result = holding.to_dict()
        
        assert result['portfolio_id'] == 1
        assert result['asset_id'] == 1
        assert result['date'] == date(2024, 1, 15)
        assert result['quantity'] == Decimal('100')
