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


class TestDataAcquisitionServiceBasics:
    """Test basic functionality of DataAcquisitionService"""

    @pytest.mark.unit
    def test_init_with_default_provider(self, db_access):
        """Test initialization with default provider"""
        service = DataAcquisitionService(db_access)
        assert isinstance(service.data_provider, YFinanceProvider)
        assert service.db_access == db_access

    @pytest.mark.unit
    def test_init_with_custom_provider(self, db_access):
        """Test initialization with custom provider"""
        test_provider = TestDataProvider()
        service = DataAcquisitionService(db_access, test_provider)
        assert service.data_provider == test_provider

    @pytest.mark.unit
    def test_switch_provider(self, db_access):
        """Test switching data providers"""
        service = DataAcquisitionService(db_access)
        original_provider = service.data_provider
        
        new_provider = TestDataProvider()
        service.switch_provider(new_provider)
        
        assert service.data_provider == new_provider
        assert service.data_provider != original_provider

    @pytest.mark.unit
    def test_fetch_asset_data_with_defaults(self, db_access):
        """Test fetching asset data with default date range"""
        test_provider = TestDataProvider()
        service = DataAcquisitionService(db_access, test_provider)
        
        asset_name, price_data = service.fetch_asset_data("TEST")
        
        assert asset_name == "Test Asset TEST"
        assert isinstance(price_data, pd.DataFrame)
        assert not price_data.empty
        assert test_provider.call_count == 1

    @pytest.mark.unit
    def test_fetch_asset_data_with_custom_dates(self, db_access):
        """Test fetching asset data with custom date range"""
        test_provider = TestDataProvider()
        service = DataAcquisitionService(db_access, test_provider)
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        asset_name, price_data = service.fetch_asset_data("TEST", start_date, end_date)
        
        assert len(price_data) == 5  # 5 days of data
        assert price_data.index[0].date() == start_date.date()
        assert price_data.index[-1].date() == end_date.date()


class TestDataPreparation:
    """Test data preparation methods"""

    @pytest.mark.unit
    def test_prepare_asset_for_db(self, db_access):
        """Test preparing asset data for database"""
        service = DataAcquisitionService(db_access)
        
        result = service.prepare_asset_for_db("AAPL", "Apple Inc.")
        
        expected = {'ticker': 'AAPL', 'name': 'Apple Inc.'}
        assert result == expected

    @pytest.mark.unit
    def test_prepare_price_history_for_db(self, db_access):
        """Test preparing price history for database"""
        service = DataAcquisitionService(db_access)
        
        # Create test DataFrame
        test_data = pd.DataFrame({
            'Date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [95.0, 96.0],
            'Close': [102.0, 103.0],
            'Volume': [1000000, 1100000]
        })
        test_data.set_index('Date', inplace=True)
        
        result = service.prepare_price_history_for_db(test_data)
        
        assert len(result) == 2
        assert result[0]['date'] == datetime(2024, 1, 1).date()
        assert result[0]['open'] == 100.0
        assert result[0]['close'] == 102.0
        assert result[1]['volume'] == 1100000

    @pytest.mark.unit
    def test_prepare_price_history_column_standardization(self, db_access):
        """Test that column names are properly standardized"""
        service = DataAcquisitionService(db_access)
        
        # Test with different column formats that might come from different providers
        test_data = pd.DataFrame({
            'Date': [datetime(2024, 1, 1)],
            'Open': [100.0],
            'High': [105.0],
            'Low': [95.0],
            'Close': [102.0],
            'Volume': [1000000]
        })
        test_data.set_index('Date', inplace=True)
        
        result = service.prepare_price_history_for_db(test_data)
        
        # Check that all expected columns are present
        expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in result[0]


class TestSpecialAssetHandling:
    """Test handling of special assets like USD and inverted currencies"""

    @pytest.mark.unit
    def test_create_usd_price_data(self, db_access):
        """Test USD price data creation"""
        service = DataAcquisitionService(db_access)
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 3)
        
        usd_data = service._create_usd_price_data(start_date, end_date)
        
        assert len(usd_data) == 3  # 3 days
        assert all(usd_data['Open'] == 1.0)
        assert all(usd_data['High'] == 1.0)
        assert all(usd_data['Low'] == 1.0)
        assert all(usd_data['Close'] == 1.0)
        assert all(usd_data['Volume'] == 0)

    @pytest.mark.unit
    def test_apply_inversion(self, db_access):
        """Test price data inversion for inverted assets"""
        service = DataAcquisitionService(db_access)
        
        # Create test data with known values
        test_data = pd.DataFrame({
            'Date': [datetime(2024, 1, 1)],
            'Open': [2.0],
            'High': [4.0],
            'Low': [1.0],
            'Close': [2.5],
            'Volume': [1000000]
        })
        test_data.set_index('Date', inplace=True)
        
        inverted_data = service._apply_inversion(test_data)
        
        assert inverted_data['Open'].iloc[0] == 0.5  # 1/2.0
        assert inverted_data['High'].iloc[0] == 0.25  # 1/4.0
        assert inverted_data['Low'].iloc[0] == 1.0  # 1/1.0
        assert inverted_data['Close'].iloc[0] == 0.4  # 1/2.5
        assert inverted_data['Volume'].iloc[0] == 1000000  # Volume unchanged


class TestErrorHandling:
    """Test error handling in data acquisition"""

    @pytest.mark.unit
    def test_provider_error_handling(self, db_access, initialized_db):
        """Test handling of provider errors"""
        # Create a provider that raises an exception
        class ErrorProvider(DataProvider):
            def fetch_asset_data(self, ticker, start_date, end_date):
                raise Exception("API Error")
            
            def get_provider_name(self):
                return "Error Provider"
        
        error_provider = ErrorProvider()
        service = DataAcquisitionService(db_access, error_provider)
        
        # Mock the get_all_assets method
        with patch.object(db_access, 'get_all_assets') as mock_get_assets:
            mock_asset = Mock()
            mock_asset.ticker = "AAPL"
            mock_asset.code = "AAPL"
            mock_asset.is_inverted = False
            mock_get_assets.return_value = [mock_asset]
            
            with patch.object(db_access, 'get_last_price_date', return_value=None):
                with patch.object(db_access, 'add_price_history') as mock_add_price:
                    # This should not raise an exception, but handle it gracefully
                    service.update_db_with_asset_data()
                    
                    # Verify add_price_history was NOT called due to error
                    mock_add_price.assert_not_called()

    @pytest.mark.unit
    def test_empty_data_handling(self, db_access, initialized_db):
        """Test handling of empty data from provider"""
        # Create a provider that returns empty data
        class EmptyDataProvider(DataProvider):
            def fetch_asset_data(self, ticker, start_date, end_date):
                empty_df = pd.DataFrame()
                return f"Asset {ticker}", empty_df
            
            def get_provider_name(self):
                return "Empty Data Provider"
        
        empty_provider = EmptyDataProvider()
        service = DataAcquisitionService(db_access, empty_provider)
        
        # Mock the get_all_assets method
        with patch.object(db_access, 'get_all_assets') as mock_get_assets:
            mock_asset = Mock()
            mock_asset.ticker = "AAPL"
            mock_asset.code = "AAPL"
            mock_asset.is_inverted = False
            mock_get_assets.return_value = [mock_asset]
            
            with patch.object(db_access, 'get_last_price_date', return_value=None):
                with patch.object(db_access, 'add_price_history') as mock_add_price:
                    service.update_db_with_asset_data()
                    
                    # Verify add_price_history was NOT called due to empty data
                    mock_add_price.assert_not_called()


class TestProviderComparison:
    """Test different providers return compatible data"""

    @pytest.mark.unit
    def test_provider_data_compatibility(self, db_access):
        """Test that different providers return compatible data formats"""
        # Create a simple mock provider for comparison
        class SimpleMockProvider(DataProvider):
            def fetch_asset_data(self, ticker, start_date, end_date):
                date_range = pd.date_range(start=start_date, end=end_date)
                price_data = pd.DataFrame({
                    'Date': date_range,
                    'Open': [50.0] * len(date_range),
                    'High': [55.0] * len(date_range),
                    'Low': [45.0] * len(date_range),
                    'Close': [52.0] * len(date_range),
                    'Volume': [500000] * len(date_range)
                })
                price_data.set_index('Date', inplace=True)
                return f"Simple Mock {ticker}", price_data
            
            def get_provider_name(self):
                return "Simple Mock Provider"
        
        providers = [
            SimpleMockProvider(),
            TestDataProvider()
        ]
        
        ticker = "TEST"
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 3)
        
        results = []
        
        for provider in providers:
            service = DataAcquisitionService(db_access, provider)
            asset_name, price_data = service.fetch_asset_data(ticker, start_date, end_date)
            
            # Prepare data for database
            price_history = service.prepare_price_history_for_db(price_data)
            
            results.append({
                'provider': provider.get_provider_name(),
                'asset_name': asset_name,
                'price_history': price_history
            })
        
        # Verify all providers return compatible data structures
        for result in results:
            assert isinstance(result['asset_name'], str)
            assert isinstance(result['price_history'], list)
            assert len(result['price_history']) > 0
            
            # Check first record structure
            first_record = result['price_history'][0]
            required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                assert field in first_record
