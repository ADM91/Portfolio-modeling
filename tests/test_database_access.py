import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from sqlalchemy.exc import IntegrityError
from database.entities import *
from database.access import DatabaseAccess


class TestDatabaseAccessBasics:
    """Test basic database access functionality"""

    @pytest.mark.unit
    def test_init_db(self, db_access, test_engine):
        """Test database initialization"""
        db_access.init_db()
        
        # Check that tables were created
        inspector = test_engine.dialect.get_table_names(test_engine.connect())
        expected_tables = ['portfolios', 'assets', 'transaction_types', 'transactions', 
                          'price_history', 'portfolio_holdings_time_series', 
                          'tax_lots', 'tax_lot_transactions']
        
        for table in expected_tables:
            assert table in inspector

    @pytest.mark.unit
    def test_insert_if_not_exists_new_record(self, db_access, db_session, sample_portfolios):
        """Test inserting new records"""
        db_access.insert_if_not_exists(db_session, Portfolio, sample_portfolios, ['name'])
        db_session.commit()
        
        portfolios = db_session.query(Portfolio).all()
        assert len(portfolios) == len(sample_portfolios)
        assert portfolios[0].name == "Test Portfolio 1"

    @pytest.mark.unit
    def test_insert_if_not_exists_duplicate_prevention(self, db_access, db_session, sample_portfolios):
        """Test that duplicates are not inserted"""
        # Insert once
        db_access.insert_if_not_exists(db_session, Portfolio, sample_portfolios, ['name'])
        db_session.commit()
        
        # Try to insert again
        db_access.insert_if_not_exists(db_session, Portfolio, sample_portfolios, ['name'])
        db_session.commit()
        
        portfolios = db_session.query(Portfolio).all()
        assert len(portfolios) == len(sample_portfolios)  # Should not have duplicates


class TestAssetManagement:
    """Test asset-related functionality"""

    @pytest.mark.unit
    def test_add_asset_new(self, db_access, db_session):
        """Test adding a new asset"""
        asset = db_access.add_asset(db_session, "AAPL", "Apple Inc.", 
                                   is_currency=False, is_inverted=False)
        db_session.commit()
        
        assert asset.ticker == "AAPL"
        assert asset.name == "Apple Inc."
        assert asset.is_currency is False
        assert asset.id is not None

    @pytest.mark.unit
    def test_add_asset_update_existing(self, db_access, db_session):
        """Test updating an existing asset"""
        # Add initial asset
        asset1 = db_access.add_asset(db_session, "AAPL", "Apple Inc.")
        db_session.commit()
        
        # Update with new name
        asset2 = db_access.add_asset(db_session, "AAPL", "Apple Inc. Updated")
        db_session.commit()
        
        assert asset1.id == asset2.id
        assert asset2.name == "Apple Inc. Updated"

    @pytest.mark.unit
    def test_get_asset_by_code(self, db_access, initialized_db):
        """Test retrieving asset by code"""
        asset = db_access.get_asset_by_code(initialized_db, "USD")
        assert asset is not None
        assert asset.code == "USD"
        assert asset.is_currency is True

    @pytest.mark.unit
    def test_get_all_assets(self, db_access, initialized_db):
        """Test retrieving all assets"""
        assets = db_access.get_all_assets(initialized_db)
        assert len(assets) >= 4  # From sample data

    @pytest.mark.unit
    def test_get_all_currencies(self, db_access, initialized_db):
        """Test retrieving only currency assets"""
        currencies = db_access.get_all_currencies(initialized_db)
        for currency in currencies:
            assert currency.is_currency is True


class TestPriceHistory:
    """Test price history functionality"""

    @pytest.mark.unit
    @pytest.mark.financial
    def test_add_price_history_new(self, db_access, initialized_db, sample_price_history):
        """Test adding new price history"""
        # Get an asset
        asset = db_access.get_asset_by_code(initialized_db, "AAPL")
        
        db_access.add_price_history(initialized_db, "AAPL", sample_price_history)
        initialized_db.commit()
        
        prices = initialized_db.query(PriceHistory).filter(
            PriceHistory.asset_id == asset.id
        ).all()
        
        assert len(prices) == len(sample_price_history)
        assert prices[0].close == Decimal('103.75')

    @pytest.mark.unit
    @pytest.mark.financial
    def test_add_price_history_update_existing(self, db_access, initialized_db, sample_price_history):
        """Test updating existing price history"""
        # Add initial data
        db_access.add_price_history(initialized_db, "AAPL", sample_price_history)
        initialized_db.commit()
        
        # Update with new close price
        updated_data = sample_price_history.copy()
        updated_data[0]['close'] = Decimal('999.99')
        
        db_access.add_price_history(initialized_db, "AAPL", updated_data)
        initialized_db.commit()
        
        asset = db_access.get_asset_by_code(initialized_db, "AAPL")
        prices = initialized_db.query(PriceHistory).filter(
            PriceHistory.asset_id == asset.id,
            PriceHistory.date == datetime(2024, 1, 1)
        ).first()
        
        assert prices.close == Decimal('999.99')

    @pytest.mark.unit
    @pytest.mark.financial
    def test_decimal_precision_preservation(self, db_access, initialized_db, high_precision_price_data):
        """Test that decimal precision is preserved (within SQLite limitations)"""
        db_access.add_price_history(initialized_db, "AAPL", high_precision_price_data)
        initialized_db.commit()
        
        asset = db_access.get_asset_by_code(initialized_db, "AAPL")
        price = initialized_db.query(PriceHistory).filter(
            PriceHistory.asset_id == asset.id
        ).first()
        
        # SQLite has precision limitations, so we test for reasonable precision
        assert str(price.close).startswith('123.98765432')
        assert isinstance(price.close, Decimal)
        # Ensure we have at least 8 decimal places of precision
        assert len(str(price.close).split('.')[1]) >= 8

    @pytest.mark.unit
    def test_get_last_price_date(self, db_access, initialized_db, sample_price_history):
        """Test getting the last price date for an asset"""
        db_access.add_price_history(initialized_db, "AAPL", sample_price_history)
        initialized_db.commit()
        
        last_date = db_access.get_last_price_date(initialized_db, "AAPL")
        assert last_date == datetime(2024, 1, 2)

    @pytest.mark.unit
    def test_get_asset_price_history_df(self, db_access, initialized_db, sample_price_history):
        """Test getting price history as DataFrame"""
        db_access.add_price_history(initialized_db, "AAPL", sample_price_history)
        initialized_db.commit()
        
        asset = db_access.get_asset_by_code(initialized_db, "AAPL")
        df = db_access.get_asset_price_history_df(
            initialized_db, asset.id, date(2024, 1, 1), date(2024, 1, 2)
        )
        
        assert len(df) == 2
        assert 'close' in df.columns
        assert df['close'].dtype.kind == 'f'  # Should be float for pandas


class TestCurrencyConversion:
    """Test currency conversion functionality"""

    @pytest.mark.unit
    @pytest.mark.financial
    def test_currency_conversion_same_currency(self, db_access, initialized_db):
        """Test conversion rate for same currency should be 1.0"""
        usd_asset = db_access.get_asset_by_code(initialized_db, "USD")
        
        rate = db_access.get_currency_conversion_on_date(
            initialized_db, usd_asset.id, usd_asset.id, datetime(2024, 1, 1)
        )
        
        assert rate == 1.0

    @pytest.mark.unit
    @pytest.mark.financial
    def test_currency_conversion_calculation(self, db_access, initialized_db, currency_conversion_test_data):
        """Test currency conversion rate calculation"""
        # Add price history for currencies
        db_access.add_price_history(initialized_db, "", currency_conversion_test_data['usd_price'])
        db_access.add_price_history(initialized_db, "EUR=X", currency_conversion_test_data['eur_price'])
        initialized_db.commit()
        
        usd_asset = db_access.get_asset_by_code(initialized_db, "USD")
        eur_asset = db_access.get_asset_by_code(initialized_db, "EUR")
        
        rate = db_access.get_currency_conversion_on_date(
            initialized_db, usd_asset.id, eur_asset.id, datetime(2024, 1, 1)
        )
        
        expected_rate = 1.0 / 0.85  # USD to EUR
        assert abs(rate - expected_rate) < 0.0001

    @pytest.mark.unit
    def test_currency_conversion_missing_data(self, db_access, initialized_db):
        """Test error handling when price data is missing"""
        usd_asset = db_access.get_asset_by_code(initialized_db, "USD")
        eur_asset = db_access.get_asset_by_code(initialized_db, "EUR")
        
        with pytest.raises(ValueError, match="Could not find price data"):
            db_access.get_currency_conversion_on_date(
                initialized_db, usd_asset.id, eur_asset.id, datetime(2024, 1, 1)
            )


class TestActionManagement:
    """Test action-related functionality"""

    @pytest.mark.unit
    def test_get_action_type_by_name(self, db_access, initialized_db):
        """Test retrieving action type by name"""
        action_type = db_access.get_action_type_by_name(initialized_db, "buy")
        assert action_type is not None
        assert action_type.name == "buy"

    @pytest.mark.unit
    def test_get_unprocessed_actions(self, db_access, initialized_db, sample_actions):
        """Test retrieving unprocessed actions"""
        # Insert sample actions
        for action_data in sample_actions:
            action = Transaction(**action_data)
            initialized_db.add(action)
        initialized_db.commit()
        
        unprocessed = db_access.get_unprocessed_actions(initialized_db)
        assert len(unprocessed) == 2
        assert all(not action.is_processed for action in unprocessed)

    @pytest.mark.unit
    def test_update_action_processed(self, db_access, initialized_db, sample_actions):
        """Test marking action as processed"""
        # Insert sample action
        action = Transaction(**sample_actions[0])
        initialized_db.add(action)
        initialized_db.commit()
        
        # Update to processed
        db_access.update_action(initialized_db, action)
        initialized_db.commit()
        
        updated_action = initialized_db.query(Transaction).filter(Transaction.id == action.id).first()
        assert updated_action.is_processed is True


class TestPortfolioManagement:
    """Test portfolio-related functionality"""

    @pytest.mark.unit
    def test_get_portfolio_by_name(self, db_access, initialized_db):
        """Test retrieving portfolio by name"""
        portfolio = db_access.get_portfolio_by_name(initialized_db, "Alexander")
        assert portfolio is not None
        assert portfolio.name == "Alexander"

    @pytest.mark.unit
    def test_get_portfolios(self, db_access, initialized_db):
        """Test retrieving all portfolios"""
        portfolios = db_access.get_portfolios(initialized_db)
        assert len(portfolios) >= 3  # From sample data

    @pytest.mark.unit
    def test_get_portfolio_assets(self, db_access, initialized_db, sample_actions):
        """Test retrieving assets for a portfolio"""
        # Insert sample actions to create portfolio-asset relationships
        for action_data in sample_actions:
            action = Transaction(**action_data)
            initialized_db.add(action)
        initialized_db.commit()
        
        assets = db_access.get_portfolio_assets(initialized_db, 1)
        assert len(assets) >= 1

    @pytest.mark.unit
    def test_get_portfolio_asset_actions(self, db_access, initialized_db, sample_actions):
        """Test retrieving actions for a specific portfolio and asset"""
        # Insert sample actions
        for action_data in sample_actions:
            action = Transaction(**action_data)
            initialized_db.add(action)
        initialized_db.commit()
        
        actions = db_access.get_portfolio_asset_actions(initialized_db, 1, 1)
        assert len(actions) == 2  # Both sample actions are for portfolio 1, asset 1


class TestHoldingsTimeSeries:
    """Test holdings time series functionality"""

    @pytest.mark.unit
    def test_store_holdings_time_series(self, db_access, initialized_db):
        """Test storing holdings time series data"""
        holdings_data = [
            {
                'portfolio_id': 1,
                'asset_id': 1,
                'date': datetime(2024, 1, 1),
                'quantity': Decimal('100.0')
            },
            {
                'portfolio_id': 1,
                'asset_id': 1,
                'date': datetime(2024, 1, 2),
                'quantity': Decimal('150.0')
            }
        ]
        
        db_access.store_holdings_time_series(initialized_db, holdings_data)
        initialized_db.commit()
        
        holdings = initialized_db.query(PortfolioHoldingsTimeSeries).all()
        assert len(holdings) == 2
        assert holdings[0].quantity == Decimal('100.0')

    @pytest.mark.unit
    def test_get_portfolio_asset_time_series_df(self, db_access, initialized_db):
        """Test getting holdings time series as DataFrame"""
        # Insert test data
        holdings_data = [
            {
                'portfolio_id': 1,
                'asset_id': 1,
                'date': datetime(2024, 1, 1),
                'quantity': Decimal('100.0')
            }
        ]
        
        db_access.store_holdings_time_series(initialized_db, holdings_data)
        initialized_db.commit()
        
        df = db_access.get_portfolio_asset_time_series_df(initialized_db, 1, 1)
        
        assert len(df) == 1
        assert 'quantity' in df.columns
        assert df['quantity'].dtype.kind == 'f'  # Should be float for pandas

    @pytest.mark.unit
    def test_get_last_holdings_time_series_update(self, db_access, initialized_db):
        """Test getting the last update date for holdings time series"""
        # Initially should be None
        last_update = db_access.get_last_holdings_time_series_update(initialized_db)
        assert last_update is None
        
        # Add some data
        holdings_data = [{
            'portfolio_id': 1,
            'asset_id': 1,
            'date': datetime(2024, 1, 1),
            'quantity': Decimal('100.0')
        }]
        
        db_access.store_holdings_time_series(initialized_db, holdings_data)
        initialized_db.commit()
        
        last_update = db_access.get_last_holdings_time_series_update(initialized_db)
        assert last_update == date(2024, 1, 1)

    @pytest.mark.unit
    def test_clear_holdings_time_series(self, db_access, initialized_db):
        """Test clearing holdings time series"""
        # Add some data first
        holdings_data = [{
            'portfolio_id': 1,
            'asset_id': 1,
            'date': datetime(2024, 1, 1),
            'quantity': Decimal('100.0')
        }]
        
        db_access.store_holdings_time_series(initialized_db, holdings_data)
        initialized_db.commit()
        
        # Verify data exists
        holdings = initialized_db.query(PortfolioHoldingsTimeSeries).all()
        assert len(holdings) == 1
        
        # Clear data
        db_access.clear_holdings_time_series(initialized_db)
        initialized_db.commit()
        
        # Verify data is cleared
        holdings = initialized_db.query(PortfolioHoldingsTimeSeries).all()
        assert len(holdings) == 0


class TestTaxLotSupport:
    """Test tax lot functionality"""

    @pytest.mark.unit
    def test_get_tax_lots_for_portfolio_asset(self, db_access, initialized_db, tax_lot_test_data):
        """Test retrieving tax lots for portfolio and asset"""
        # Insert test tax lots
        for lot_data in tax_lot_test_data['lots']:
            lot = TaxLot(
                portfolio_id=tax_lot_test_data['portfolio_id'],
                asset_id=tax_lot_test_data['asset_id'],
                currency_id=tax_lot_test_data['currency_id'],
                **lot_data
            )
            initialized_db.add(lot)
        initialized_db.commit()
        
        lots = db_access.get_tax_lots_for_portfolio_asset(
            initialized_db, 
            tax_lot_test_data['portfolio_id'], 
            tax_lot_test_data['asset_id']
        )
        
        assert len(lots) == 2
        assert lots[0].lot_identifier == 'LOT_001'
        assert lots[0].remaining_quantity == Decimal('100.0')

    @pytest.mark.unit
    def test_get_tax_lots_include_closed(self, db_access, initialized_db, tax_lot_test_data):
        """Test retrieving tax lots including closed ones"""
        # Insert test tax lots with one closed
        lot_data = tax_lot_test_data['lots'][0].copy()
        lot_data['is_closed'] = True
        
        lot = TaxLot(
            portfolio_id=tax_lot_test_data['portfolio_id'],
            asset_id=tax_lot_test_data['asset_id'],
            currency_id=tax_lot_test_data['currency_id'],
            **lot_data
        )
        initialized_db.add(lot)
        initialized_db.commit()
        
        # Test excluding closed (default)
        lots_open = db_access.get_tax_lots_for_portfolio_asset(
            initialized_db, 
            tax_lot_test_data['portfolio_id'], 
            tax_lot_test_data['asset_id']
        )
        assert len(lots_open) == 0
        
        # Test including closed
        lots_all = db_access.get_tax_lots_for_portfolio_asset(
            initialized_db, 
            tax_lot_test_data['portfolio_id'], 
            tax_lot_test_data['asset_id'],
            include_closed=True
        )
        assert len(lots_all) == 1


class TestFinancialFieldConversion:
    """Test financial field conversion functionality"""

    @pytest.mark.unit
    @pytest.mark.financial
    def test_convert_financial_fields_to_decimal(self, db_access):
        """Test conversion of financial fields to Decimal"""
        test_data = {
            'price': 123.45,
            'quantity': '67.89',
            'fee': 9.99,
            'non_financial_field': 'test_value'
        }
        
        converted = db_access._convert_financial_fields_to_decimal(test_data, Transaction)
        
        assert isinstance(converted['price'], Decimal)
        assert isinstance(converted['quantity'], Decimal)
        assert isinstance(converted['fee'], Decimal)
        assert converted['non_financial_field'] == 'test_value'
        
        assert str(converted['price']) == '123.45'
        assert str(converted['quantity']) == '67.89'

    @pytest.mark.unit
    @pytest.mark.financial
    def test_convert_financial_fields_invalid_values(self, db_access):
        """Test handling of invalid financial field values"""
        test_data = {
            'price': 'invalid_price',
            'quantity': None,
            'fee': 9.99
        }
        
        # Should not raise exception, but log warning
        converted = db_access._convert_financial_fields_to_decimal(test_data, Transaction)
        
        assert converted['price'] == 'invalid_price'  # Should remain unchanged
        assert converted['quantity'] is None
        assert isinstance(converted['fee'], Decimal)


class TestErrorHandling:
    """Test error handling scenarios"""

    @pytest.mark.unit
    def test_add_price_history_nonexistent_asset(self, db_access, db_session, sample_price_history):
        """Test error when adding price history for non-existent asset"""
        with pytest.raises(ValueError, match="Asset with ticker NONEXISTENT not found"):
            db_access.add_price_history(db_session, "NONEXISTENT", sample_price_history)

    @pytest.mark.unit
    def test_get_last_price_date_nonexistent_asset(self, db_access, db_session):
        """Test getting last price date for non-existent asset"""
        result = db_access.get_last_price_date(db_session, "NONEXISTENT")
        assert result is None
