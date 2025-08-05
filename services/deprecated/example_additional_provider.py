"""
Example of how to add additional data providers to the system.
This file demonstrates the extensibility of the new architecture.
"""

from datetime import datetime
from typing import Tuple
import pandas as pd
from services.data_providers import DataProvider


class MockDataProvider(DataProvider):
    """
    Mock data provider for testing or demonstration purposes.
    Generates random price data instead of fetching from an API.
    """
    
    def __init__(self):
        import random
        self.random = random
    
    def fetch_asset_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Tuple[str, pd.DataFrame]:
        """
        Generate mock asset data for testing.
        
        :param ticker: The asset ticker symbol
        :param start_date: Start date for historical data
        :param end_date: End date for historical data
        :return: Tuple containing (asset_name, price_data_dataframe)
        """
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
    
    def get_provider_name(self) -> str:
        return "Mock Data Provider"


# Example usage:
if __name__ == "__main__":
    from database.access import DatabaseAccess
    from services.data_acquisition_service import DataAcquisitionService
    
    # Create service with mock provider
    mock_provider = MockDataProvider()
    service = DataAcquisitionService(DatabaseAccess(), mock_provider)
    
    # Test fetching data
    asset_name, price_data = service.fetch_asset_data("TEST", 
                                                     datetime(2024, 1, 1), 
                                                     datetime(2024, 1, 10))
    
    print(f"Asset: {asset_name}")
    print(f"Data shape: {price_data.shape}")
    print(price_data.head())


# Here's how you would add a real API provider:
"""
class AlphaVantageProvider(DataProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_asset_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Tuple[str, pd.DataFrame]:
        import requests
        
        # Alpha Vantage API call
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        # Process the response and convert to standardized DataFrame format
        # ... implementation details ...
        
        return asset_name, price_dataframe
    
    def get_provider_name(self) -> str:
        return "Alpha Vantage"

# Usage:
# alpha_provider = AlphaVantageProvider("your_api_key")
# service = DataAcquisitionService(DatabaseAccess(), alpha_provider)
"""
