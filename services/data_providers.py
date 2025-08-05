from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple, Optional
import pandas as pd


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def fetch_asset_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Tuple[str, pd.DataFrame]:
        """
        Fetch asset data from the provider.
        
        :param ticker: The asset ticker symbol
        :param start_date: Start date for historical data
        :param end_date: End date for historical data
        :return: Tuple containing (asset_name, price_data_dataframe)
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the data provider."""
        pass


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider implementation."""
    
    def __init__(self):
        import yfinance as yf
        self.yf = yf
    
    def fetch_asset_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Tuple[str, pd.DataFrame]:
        """
        Fetch asset data from Yahoo Finance.
        
        :param ticker: The stock ticker symbol
        :param start_date: Start date for historical data
        :param end_date: End date for historical data
        :return: Tuple containing (asset_name, price_data_dataframe)
        """
        yf_ticker = self.yf.Ticker(ticker)
        
        # Fetch asset info
        info = yf_ticker.info
        asset_name = info.get('longName', info.get('shortName', ticker))
        
        # Fetch price history data
        price_data = yf_ticker.history(start=start_date, end=end_date)
        
        # Format price data
        price_data = price_data.round(2)
        
        return asset_name, price_data
    
    def get_provider_name(self) -> str:
        return "Yahoo Finance"


# Future providers can be added here, for example:
# class AlphaVantageProvider(DataProvider):
#     def fetch_asset_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Tuple[str, pd.DataFrame]:
#         # Implementation for Alpha Vantage API
#         pass
#     
#     def get_provider_name(self) -> str:
#         return "Alpha Vantage"
