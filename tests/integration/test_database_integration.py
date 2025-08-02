import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from database.entities_improved import *
from database.access_improved import DatabaseAccess


class TestDatabaseIntegration:
    """Integration tests for complete database workflows"""

    @pytest.mark.integration
    def test_complete_action_workflow(self, db_access, initialized_db):
        """Test complete workflow: Add asset → Add price history → Create action → Update holdings"""
        
        # Step 1: Add asset
        asset = db_access.add_asset(initialized_db, "TSLA", "Tesla Inc.", is_currency=False)
        usd_asset = db_access.get_asset_by_code(initialized_db, "USD")
        portfolio = db_access.get_portfolio_by_name(initialized_db, "Alexander")
        buy_action_type = db_access.get_action_type_by_name(initialized_db, "buy")
        initialized_db.commit()
        
        # Step 2: Add price history
        price_history = [
            {
                'date': datetime(2024, 1, 1),
                'open': Decimal('200.00'),
                'high': Decimal('210.00'),
                'low': Decimal('195.00'),
                'close': Decimal('205.00'),
                'volume': 1000000
            },
            {
                'date': datetime(2024, 1, 2),
                'open': Decimal('205.00'),
                'high': Decimal('215.00'),
                'low': Decimal('200.00'),
                'close': Decimal('210.00'),
                'volume': 1200000
            }
        ]
        
        db_access.add_price_history(initialized_db, "TSLA", price_history)
        initialized_db.commit()
        
        # Step 3: Create buy action
        action = Transaction(
            portfolio_id=portfolio.id,
            action_type_id=buy_action_type.id,
            transaction_datetime=datetime(2024, 1, 1),
            asset_id=asset.id,
            currency_id=usd_asset.id,
            price=Decimal('205.00'),
            quantity=Decimal('10.0'),
            fee=Decimal('9.99'),
            platform='Test Broker',
            comment='Integration test buy',
            is_processed=False
        )
        initialized_db.add(action)
        initialized_db.commit()
        
        # Step 4: Update holdings time series
        holdings_data = [{
            'portfolio_id': portfolio.id,
            'asset_id': asset.id,
            'date': datetime(2024, 1, 1),
            'quantity': Decimal('10.0')
        }]
        
        db_access.store_holdings_time_series(initialized_db, holdings_data)
        initialized_db.commit()
        
        # Verify complete workflow
        # Check asset exists
        retrieved_asset = db_access.get_asset_by_code(initialized_db, "TSLA")
        assert retrieved_asset is not None
        assert retrieved_asset.name == "Tesla Inc."
        
        # Check price history exists
        last_price_date = db_access.get_last_price_date(initialized_db, "TSLA")
        assert last_price_date == datetime(2024, 1, 2)
        
        # Check action exists
        actions = db_access.get_portfolio_asset_actions(initialized_db, portfolio.id, asset.id)
        assert len(actions) == 1
        assert actions[0].quantity == Decimal('10.0')
        
        # Check holdings time series exists
        holdings = db_access.get_portfolio_asset_time_series(initialized_db, portfolio.id, asset.id)
        assert len(holdings) == 1
        assert holdings[0].quantity == Decimal('10.0')

    @pytest.mark.integration
    def test_buy_sell_workflow_with_tax_lots(self, db_access, initialized_db):
        """Test buy/sell workflow with tax lot creation and tracking"""
        
        # Setup
        asset = db_access.add_asset(initialized_db, "NVDA", "NVIDIA Corp.", is_currency=False)
        usd_asset = db_access.get_asset_by_code(initialized_db, "USD")
        portfolio = db_access.get_portfolio_by_name(initialized_db, "Alexander")
        buy_action_type = db_access.get_action_type_by_name(initialized_db, "buy")
        sell_action_type = db_access.get_action_type_by_name(initialized_db, "sell")
        initialized_db.commit()
        
        # Add price history
        price_history = [
            {
                'date': datetime(2024, 1, 1),
                'open': Decimal('400.00'),
                'high': Decimal('420.00'),
                'low': Decimal('390.00'),
                'close': Decimal('410.00'),
                'volume': 500000
            },
            {
                'date': datetime(2024, 2, 1),
                'open': Decimal('410.00'),
                'high': Decimal('450.00'),
                'low': Decimal('405.00'),
                'close': Decimal('445.00'),
                'volume': 600000
            }
        ]
        
        db_access.add_price_history(initialized_db, "NVDA", price_history)
        initialized_db.commit()
        
        # Create buy action
        buy_action = Transaction(
            portfolio_id=portfolio.id,
            action_type_id=buy_action_type.id,
            transaction_datetime=datetime(2024, 1, 1),
            asset_id=asset.id,
            currency_id=usd_asset.id,
            price=Decimal('410.00'),
            quantity=Decimal('100.0'),
            fee=Decimal('9.99'),
            platform='Test Broker',
            comment='Buy for tax lot test',
            is_processed=False
        )
        initialized_db.add(buy_action)
        initialized_db.commit()
        
        # Create tax lot for buy action
        tax_lot = TaxLot(
            portfolio_id=portfolio.id,
            asset_id=asset.id,
            currency_id=usd_asset.id,
            lot_identifier=f'LOT_{buy_action.id}',
            acquisition_date=buy_action.transaction_datetime,
            original_quantity=buy_action.quantity,
            remaining_quantity=buy_action.quantity,
            cost_basis_per_unit=buy_action.price,
            is_closed=False
        )
        initialized_db.add(tax_lot)
        initialized_db.commit()
        
        # Create sell action
        sell_action = Transaction(
            portfolio_id=portfolio.id,
            action_type_id=sell_action_type.id,
            transaction_datetime=datetime(2024, 2, 1),
            asset_id=asset.id,
            currency_id=usd_asset.id,
            price=Decimal('445.00'),
            quantity=Decimal('50.0'),
            fee=Decimal('9.99'),
            platform='Test Broker',
            comment='Sell for tax lot test',
            is_processed=False
        )
        initialized_db.add(sell_action)
        initialized_db.commit()
        
        # Create tax lot transaction for sell
        proceeds = sell_action.quantity * sell_action.price - sell_action.fee
        cost_basis = sell_action.quantity * tax_lot.cost_basis_per_unit
        realized_gain_loss = proceeds - cost_basis
        
        tax_lot_transaction = TaxLotTransaction(
            action_id=sell_action.id,
            tax_lot_id=tax_lot.id,
            quantity_change=-sell_action.quantity,
            proceeds=proceeds,
            realized_gain_loss=realized_gain_loss
        )
        initialized_db.add(tax_lot_transaction)
        
        # Update tax lot remaining quantity
        tax_lot.remaining_quantity = tax_lot.remaining_quantity - sell_action.quantity
        initialized_db.commit()
        
        # Verify workflow
        # Check actions exist
        actions = db_access.get_portfolio_asset_actions(initialized_db, portfolio.id, asset.id)
        assert len(actions) == 2
        
        # Check tax lot exists and is updated
        tax_lots = db_access.get_tax_lots_for_portfolio_asset(initialized_db, portfolio.id, asset.id)
        assert len(tax_lots) == 1
        assert tax_lots[0].remaining_quantity == Decimal('50.0')
        
        # Check tax lot transaction exists
        transactions = db_access.get_tax_lot_transactions_for_action(initialized_db, sell_action.id)
        assert len(transactions) == 1
        assert transactions[0].quantity_change == Decimal('-50.0')
        assert transactions[0].realized_gain_loss > 0  # Should be a gain

    @pytest.mark.integration
    def test_multi_currency_portfolio_workflow(self, db_access, initialized_db, currency_conversion_test_data):
        """Test workflow with multiple currencies and conversions"""
        
        # Setup assets and portfolio
        eur_stock = db_access.add_asset(initialized_db, "SAP", "SAP SE", is_currency=False)
        usd_asset = db_access.get_asset_by_code(initialized_db, "USD")
        eur_asset = db_access.get_asset_by_code(initialized_db, "EUR")
        portfolio = db_access.get_portfolio_by_name(initialized_db, "Alexander")
        buy_action_type = db_access.get_action_type_by_name(initialized_db, "buy")
        initialized_db.commit()
        
        # Add currency price history
        db_access.add_price_history(initialized_db, "", currency_conversion_test_data['usd_price'])
        db_access.add_price_history(initialized_db, "EUR=X", currency_conversion_test_data['eur_price'])
        initialized_db.commit()
        
        # Add stock price history in EUR
        stock_price_history = [
            {
                'date': datetime(2024, 1, 1),
                'open': Decimal('100.00'),
                'high': Decimal('105.00'),
                'low': Decimal('98.00'),
                'close': Decimal('102.50'),
                'volume': 100000
            }
        ]
        
        db_access.add_price_history(initialized_db, "SAP", stock_price_history)
        initialized_db.commit()
        
        # Create buy action in EUR
        buy_action = Transaction(
            portfolio_id=portfolio.id,
            action_type_id=buy_action_type.id,
            transaction_datetime=datetime(2024, 1, 1),
            asset_id=eur_stock.id,
            currency_id=eur_asset.id,  # Transaction in EUR
            price=Decimal('102.50'),
            quantity=Decimal('10.0'),
            fee=Decimal('5.00'),
            platform='European Broker',
            comment='Multi-currency test',
            is_processed=False
        )
        initialized_db.add(buy_action)
        initialized_db.commit()
        
        # Test currency conversion
        conversion_rate = db_access.get_currency_conversion_on_date(
            initialized_db, eur_asset.id, usd_asset.id, datetime(2024, 1, 1)
        )
        
        # Should be EUR to USD conversion (0.85 EUR = 1 USD, so 1 EUR = 0.85 USD)
        expected_rate = 0.85
        assert abs(conversion_rate - expected_rate) < 0.0001
        
        # Calculate USD equivalent values
        usd_price = buy_action.price * Decimal(str(conversion_rate))
        usd_total_cost = (buy_action.price * buy_action.quantity + buy_action.fee) * Decimal(str(conversion_rate))
        
        # Verify calculations
        assert usd_price < buy_action.price  # USD price should be lower than EUR price (since EUR is stronger)
        assert usd_total_cost > Decimal('800.00')  # Should be reasonable USD amount
        
        # Verify action and relationships
        actions = db_access.get_portfolio_asset_actions(initialized_db, portfolio.id, eur_stock.id)
        assert len(actions) == 1
        assert actions[0].currency.code == "EUR"
        assert actions[0].asset.ticker == "SAP"

    @pytest.mark.integration
    def test_dividend_workflow(self, db_access, initialized_db):
        """Test dividend action workflow"""
        
        # Setup
        asset = db_access.add_asset(initialized_db, "KO", "Coca-Cola Co.", is_currency=False)
        usd_asset = db_access.get_asset_by_code(initialized_db, "USD")
        portfolio = db_access.get_portfolio_by_name(initialized_db, "Alexander")
        buy_action_type = db_access.get_action_type_by_name(initialized_db, "buy")
        dividend_action_type = db_access.get_action_type_by_name(initialized_db, "dividend")
        initialized_db.commit()
        
        # Create initial buy action
        buy_action = Transaction(
            portfolio_id=portfolio.id,
            action_type_id=buy_action_type.id,
            transaction_datetime=datetime(2024, 1, 1),
            asset_id=asset.id,
            currency_id=usd_asset.id,
            price=Decimal('60.00'),
            quantity=Decimal('100.0'),
            fee=Decimal('9.99'),
            platform='Test Broker',
            comment='Initial position for dividend test',
            is_processed=False
        )
        initialized_db.add(buy_action)
        initialized_db.commit()
        
        # Create dividend action
        # Dividend: $0.44 per share for 100 shares = $44.00 total
        dividend_action = Transaction(
            portfolio_id=portfolio.id,
            action_type_id=dividend_action_type.id,
            transaction_datetime=datetime(2024, 3, 15),  # Typical dividend payment date
            asset_id=asset.id,
            currency_id=usd_asset.id,
            price=Decimal('0.44'),  # Dividend per share
            quantity=Decimal('100.0'),  # Number of shares
            fee=Decimal('0.00'),  # No fee for dividends
            platform='Test Broker',
            comment='Quarterly dividend payment',
            is_processed=False
        )
        initialized_db.add(dividend_action)
        initialized_db.commit()
        
        # Update holdings time series to reflect dividend (cash position)
        # Note: In practice, dividends might be reinvested or held as cash
        # For this test, we'll treat it as a cash dividend that doesn't change share count
        holdings_data = [
            {
                'portfolio_id': portfolio.id,
                'asset_id': asset.id,
                'date': datetime(2024, 1, 1),
                'quantity': Decimal('100.0')  # Initial shares
            },
            {
                'portfolio_id': portfolio.id,
                'asset_id': asset.id,
                'date': datetime(2024, 3, 15),
                'quantity': Decimal('100.0')  # Same shares after dividend
            }
        ]
        
        db_access.store_holdings_time_series(initialized_db, holdings_data)
        initialized_db.commit()
        
        # Verify workflow
        actions = db_access.get_portfolio_asset_actions(initialized_db, portfolio.id, asset.id)
        assert len(actions) == 2
        
        # Find dividend action
        dividend_actions = [a for a in actions if a.transaction_type.name == "dividend"]
        assert len(dividend_actions) == 1
        
        dividend = dividend_actions[0]
        assert dividend.price == Decimal('0.44')
        assert dividend.quantity == Decimal('100.0')
        
        # Calculate total dividend received
        total_dividend = dividend.price * dividend.quantity
        assert total_dividend == Decimal('44.00')
        
        # Verify holdings remain the same (cash dividend doesn't change share count)
        holdings = db_access.get_portfolio_asset_time_series(initialized_db, portfolio.id, asset.id)
        assert len(holdings) == 2
        assert all(h.quantity == Decimal('100.0') for h in holdings)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_bulk_operations(self, db_access, initialized_db):
        """Test performance of bulk operations with larger datasets"""
        
        # Setup
        asset = db_access.add_asset(initialized_db, "BULK", "Bulk Test Asset", is_currency=False)
        usd_asset = db_access.get_asset_by_code(initialized_db, "USD")
        portfolio = db_access.get_portfolio_by_name(initialized_db, "Alexander")
        initialized_db.commit()
        
        # Generate large price history dataset (1 year of daily data)
        import pandas as pd
        date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        price_history = []
        base_price = Decimal('100.00')
        
        for i, date in enumerate(date_range):
            # Simulate price movement
            price_change = Decimal(str((i % 10 - 5) * 0.01))  # Small daily changes
            current_price = base_price + price_change
            
            price_history.append({
                'date': date.to_pydatetime(),
                'open': current_price - Decimal('0.50'),
                'high': current_price + Decimal('1.00'),
                'low': current_price - Decimal('1.00'),
                'close': current_price,
                'volume': 1000000 + (i * 1000)
            })
        
        # Test bulk price history insertion
        import time
        start_time = time.time()
        
        db_access.add_price_history(initialized_db, "BULK", price_history)
        initialized_db.commit()
        
        bulk_insert_time = time.time() - start_time
        
        # Verify all data was inserted
        prices = initialized_db.query(PriceHistory).filter(
            PriceHistory.asset_id == asset.id
        ).count()
        
        assert prices == len(price_history)
        assert bulk_insert_time < 5.0  # Should complete within 5 seconds
        
        # Test bulk holdings time series insertion
        holdings_data = []
        for i, date in enumerate(date_range[:30]):  # First 30 days
            holdings_data.append({
                'portfolio_id': portfolio.id,
                'asset_id': asset.id,
                'date': date.to_pydatetime(),
                'quantity': Decimal('100.0') + Decimal(str(i))  # Gradually increasing
            })
        
        start_time = time.time()
        db_access.store_holdings_time_series(initialized_db, holdings_data)
        initialized_db.commit()
        bulk_holdings_time = time.time() - start_time
        
        # Verify holdings data
        holdings_count = initialized_db.query(PortfolioHoldingsTimeSeries).filter(
            PortfolioHoldingsTimeSeries.portfolio_id == portfolio.id,
            PortfolioHoldingsTimeSeries.asset_id == asset.id
        ).count()
        
        assert holdings_count == len(holdings_data)
        assert bulk_holdings_time < 2.0  # Should complete within 2 seconds
