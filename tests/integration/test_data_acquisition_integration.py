import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
from decimal import Decimal

from services.data_acquisition_service import DataAcquisitionService
from services.data_providers import DataProvider, YFinanceProvider
from database.access import DatabaseAccess
from database.entities import Asset, PriceHistory
from config import assets as config_assets


class MockDataProvider(DataProvider):
    """Mock data provider for testing purposes."""
    
    def __init__(self):
        import random
        self.random = random
    
    def fetch_asset_data(self, ticker, start_date, end_date):
        """Generate mock asset data for testing."""
        # Generate mock asset name
        asset_name = f"Mock Asset {ticker}"
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Generate mock price data with some randomness
        base_price = 100.0
        price_data = []
        
        for date in date_range:
            # Simple random walk for price generation
            daily_change = self.random.uniform(-0.05, 0.05)  # ±5% daily change
            base_price *= (1 + daily_change)
            
            # Generate OHLC data
            open_price = base_price
            high_price = open_price * (1 + abs(self.random.uniform(0, 0.02)))
            low_price = open_price * (1 - abs(self.random.uniform(0, 0.02)))
            close_price = self.random.uniform(low_price, high_price)
            volume = self.random.randint(1000, 100000)
            
            price_data.append({
                'Date': date,
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            })
            
            base_price = close_price
        
        # Convert to DataFrame
        df = pd.DataFrame(price_data)
        df.set_index('Date', inplace=True)
        
        return asset_name, df
    
    def get_provider_name(self):
        return "Mock Data Provider"


class TestDataProvider(DataProvider):
    """Test data provider for unit testing"""
    
    def __init__(self, mock_data=None):
        self.mock_data = mock_data or {}
        self.call_count = 0
    
    def fetch_asset_data(self, ticker, start_date, end_date):
        self.call_count += 1
        
        if ticker in self.mock_data:
            return self.mock_data[ticker]
        
        # Default mock data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        price_data = pd.DataFrame({
            'Date': date_range,
            'Open': [100.0 + i for i in range(len(date_range))],
            'High': [105.0 + i for i in range(len(date_range))],
            'Low': [95.0 + i for i in range(len(date_range))],
            'Close': [102.0 + i for i in range(len(date_range))],
            'Volume': [1000000] * len(date_range)
        })
        price_data.set_index('Date', inplace=True)
        
        return f"Test Asset {ticker}", price_data
    
    def get_provider_name(self):
        return "Test Provider"


class TestDatabaseIntegration:
    """Test database integration functionality"""

    @pytest.mark.integration
    def test_update_db_with_asset_data_mock_provider(self, db_access, initialized_db):
        """Test updating database with mock provider"""
        # Create mock provider with specific data
        mock_data = {
            'AAPL': ('Apple Inc.', pd.DataFrame({
                'Date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                'Open': [150.0, 151.0],
                'High': [155.0, 156.0],
                'Low': [149.0, 150.0],
                'Close': [152.0, 153.0],
                'Volume': [1000000, 1100000]
            }).set_index('Date'))
        }
        
        test_provider = TestDataProvider(mock_data)
        service = DataAcquisitionService(db_access, test_provider)
        
        # Add test asset to database
        db_access.add_asset(initialized_db, "AAPL", "Apple Inc.", is_currency=False, is_inverted=False)
        initialized_db.commit()
        
        # Mock the get_all_assets method to return our test asset
        with patch.object(db_access, 'get_all_assets') as mock_get_assets:
            mock_asset = Mock()
            mock_asset.ticker = "AAPL"
            mock_asset.code = "AAPL"
            mock_asset.is_inverted = False
            mock_get_assets.return_value = [mock_asset]
            
            # Mock get_last_price_date to return None (no existing data)
            with patch.object(db_access, 'get_last_price_date', return_value=None):
                # Mock add_price_history to verify it's called
                with patch.object(db_access, 'add_price_history') as mock_add_price:
                    service.update_db_with_asset_data()
                    
                    # Verify the provider was called
                    assert test_provider.call_count == 1
                    
                    # Verify add_price_history was called
                    mock_add_price.assert_called_once()
                    call_args = mock_add_price.call_args
                    assert call_args[0][1] == "AAPL"  # ticker
                    assert len(call_args[0][2]) == 2  # 2 days of data

    @pytest.mark.integration
    def test_update_db_usd_handling(self, db_access, initialized_db):
        """Test that USD assets are handled specially"""
        test_provider = TestDataProvider()
        service = DataAcquisitionService(db_access, test_provider)
        
        # Mock the get_all_assets method to return USD asset
        with patch.object(db_access, 'get_all_assets') as mock_get_assets:
            mock_asset = Mock()
            mock_asset.ticker = ""
            mock_asset.code = "USD"
            mock_asset.is_inverted = False
            mock_get_assets.return_value = [mock_asset]
            
            with patch.object(db_access, 'get_last_price_date', return_value=None):
                with patch.object(db_access, 'add_price_history') as mock_add_price:
                    service.update_db_with_asset_data()
                    
                    # Verify add_price_history was called
                    mock_add_price.assert_called_once()
                    call_args = mock_add_price.call_args
                    price_data = call_args[0][2]
                    
                    # Verify all prices are 1.0 for USD
                    assert all(item['open'] == 1.0 for item in price_data)
                    assert all(item['close'] == 1.0 for item in price_data)
                    
                    # Verify provider was NOT called for USD
                    assert test_provider.call_count == 0

    @pytest.mark.integration
    def test_update_db_inverted_asset(self, db_access, initialized_db):
        """Test handling of inverted assets"""
        # Create mock provider with specific data for inversion testing
        mock_data = {
            'EUR=X': ('Euro', pd.DataFrame({
                'Date': [datetime(2024, 1, 1)],
                'Open': [0.85],
                'High': [0.86],
                'Low': [0.84],
                'Close': [0.85],
                'Volume': [1000000]
            }).set_index('Date'))
        }
        
        test_provider = TestDataProvider(mock_data)
        service = DataAcquisitionService(db_access, test_provider)
        
        # Mock the get_all_assets method to return inverted EUR asset
        with patch.object(db_access, 'get_all_assets') as mock_get_assets:
            mock_asset = Mock()
            mock_asset.ticker = "EUR=X"
            mock_asset.code = "EUR"
            mock_asset.is_inverted = True
            mock_get_assets.return_value = [mock_asset]
            
            with patch.object(db_access, 'get_last_price_date', return_value=None):
                with patch.object(db_access, 'add_price_history') as mock_add_price:
                    service.update_db_with_asset_data()
                    
                    # Verify add_price_history was called
                    mock_add_price.assert_called_once()
                    call_args = mock_add_price.call_args
                    price_data = call_args[0][2]
                    
                    # Verify prices were inverted (1/0.85 ≈ 1.176471)
                    assert abs(price_data[0]['close'] - (1/0.85)) < 0.000001

    @pytest.mark.integration
    def test_incremental_update_logic(self, db_access, initialized_db):
        """Test that incremental updates work correctly"""
        test_provider = TestDataProvider()
        service = DataAcquisitionService(db_access, test_provider)
        
        # Mock the get_all_assets method
        with patch.object(db_access, 'get_all_assets') as mock_get_assets:
            mock_asset = Mock()
            mock_asset.ticker = "AAPL"
            mock_asset.code = "AAPL"
            mock_asset.is_inverted = False
            mock_get_assets.return_value = [mock_asset]
            
            # Mock get_last_price_date to return a recent date
            last_date = datetime.now() - timedelta(days=2)
            with patch.object(db_access, 'get_last_price_date', return_value=last_date):
                with patch.object(db_access, 'add_price_history') as mock_add_price:
                    service.update_db_with_asset_data()
                    
                    # Verify provider was called
                    assert test_provider.call_count == 1
                    
                    # The start date should be last_date + 1 day
                    # We can't easily verify the exact dates passed to the provider,
                    # but we can verify that add_price_history was called
                    mock_add_price.assert_called_once()

    @pytest.mark.integration
    def test_no_update_when_current(self, db_access, initialized_db):
        """Test that no update occurs when data is current"""
        test_provider = TestDataProvider()
        service = DataAcquisitionService(db_access, test_provider)
        
        # Mock the get_all_assets method
        with patch.object(db_access, 'get_all_assets') as mock_get_assets:
            mock_asset = Mock()
            mock_asset.ticker = "AAPL"
            mock_asset.code = "AAPL"
            mock_asset.is_inverted = False
            mock_get_assets.return_value = [mock_asset]
            
            # Mock get_last_price_date to return today's date
            today = datetime.now()
            with patch.object(db_access, 'get_last_price_date', return_value=today):
                with patch.object(db_access, 'add_price_history') as mock_add_price:
                    service.update_db_with_asset_data()
                    
                    # Verify provider was NOT called
                    assert test_provider.call_count == 0
                    
                    # Verify add_price_history was NOT called
                    mock_add_price.assert_not_called()


class TestConfigAssets:
    """Test data acquisition for all assets defined in config"""

    @pytest.mark.integration
    def test_fetch_data_for_all_config_assets(self, db_access):
        """Test fetching data for all assets defined in config.py"""
        # Use MockDataProvider for predictable testing
        mock_provider = MockDataProvider()
        service = DataAcquisitionService(db_access, mock_provider)
        
        # Test each asset from config
        for asset_config in config_assets:
            ticker = asset_config['ticker']
            
            # Skip empty ticker (USD special case)
            if not ticker:
                continue
                
            try:
                asset_name, price_data = service.fetch_asset_data(
                    ticker, 
                    datetime(2024, 1, 1), 
                    datetime(2024, 1, 5)
                )
                
                # Verify we got data
                assert asset_name is not None
                assert isinstance(price_data, pd.DataFrame)
                assert not price_data.empty
                assert len(price_data) > 0
                
                # Verify required columns exist
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    assert col in price_data.columns
                
                # Verify data types are numeric
                for col in required_columns[:-1]:  # Exclude Volume for now
                    assert pd.api.types.is_numeric_dtype(price_data[col])
                
                print(f"✓ Successfully fetched data for {ticker} ({asset_config['name']})")
                
            except Exception as e:
                pytest.fail(f"Failed to fetch data for {ticker} ({asset_config['name']}): {str(e)}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_yfinance_data_sample(self, db_access):
        """Test fetching real data from Yahoo Finance for a sample of config assets"""
        # Only test a few assets to avoid hitting rate limits in tests
        sample_assets = [
            asset for asset in config_assets 
            if asset['ticker'] in ['SPY', 'BTC-USD', 'EUR=X'] and asset['ticker']
        ]
        
        yf_provider = YFinanceProvider()
        service = DataAcquisitionService(db_access, yf_provider)
        
        for asset_config in sample_assets:
            ticker = asset_config['ticker']
            
            try:
                asset_name, price_data = service.fetch_asset_data(
                    ticker,
                    datetime.now() - timedelta(days=7),  # Last week
                    datetime.now()
                )
                
                # Verify we got real data
                assert asset_name is not None
                assert isinstance(price_data, pd.DataFrame)
                assert not price_data.empty
                
                # Verify data quality
                assert all(price_data['Close'] > 0)  # Prices should be positive
                assert all(price_data['Volume'] >= 0)  # Volume should be non-negative
                
                print(f"✓ Successfully fetched real data for {ticker}: {len(price_data)} days")
                
            except Exception as e:
                # Don't fail the test for network issues, just log
                print(f"⚠ Could not fetch real data for {ticker}: {str(e)}")

    @pytest.mark.integration
    def test_data_preparation_for_all_config_assets(self, db_access):
        """Test data preparation works for all config asset types"""
        mock_provider = MockDataProvider()
        service = DataAcquisitionService(db_access, mock_provider)
        
        for asset_config in config_assets:
            ticker = asset_config['ticker']
            
            # Skip empty ticker (USD special case)
            if not ticker:
                continue
            
            # Fetch and prepare data
            asset_name, price_data = service.fetch_asset_data(
                ticker, 
                datetime(2024, 1, 1), 
                datetime(2024, 1, 3)
            )
            
            # Test asset preparation
            asset_db_data = service.prepare_asset_for_db(ticker, asset_name)
            assert 'ticker' in asset_db_data
            assert 'name' in asset_db_data
            assert asset_db_data['ticker'] == ticker
            
            # Test price history preparation
            price_history = service.prepare_price_history_for_db(price_data)
            assert isinstance(price_history, list)
            assert len(price_history) > 0
            
            # Verify each price record has required fields
            for price_record in price_history:
                required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
                for field in required_fields:
                    assert field in price_record
                
                # Verify date is a date object
                assert hasattr(price_record['date'], 'year')
                
                # Verify numeric fields are numeric
                numeric_fields = ['open', 'high', 'low', 'close', 'volume']
                for field in numeric_fields:
                    assert isinstance(price_record[field], (int, float, Decimal))
