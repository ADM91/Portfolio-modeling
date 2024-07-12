
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from database.access import DatabaseAccess


class YFinanceService:
    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access

    def fetch_asset_data(self, ticker: str, start_date: datetime = None, end_date: datetime = None):
        """
        Fetches asset data from yFinance API.
        
        :param ticker: The stock ticker symbol
        :param start_date: Start date for historical data (default: 5 years ago)
        :param end_date: End date for historical data (default: today)
        :return: Tuple containing asset info and price history
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=5*365)  # 5 years ago
        if not end_date:
            end_date = datetime.now()

        yf_ticker = yf.Ticker(ticker)
        
        # Fetch asset info
        info = yf_ticker.info
        asset_name = info.get('longName', info.get('shortName', ticker))
        
        # Fetch price data
        price_data = yf_ticker.history(start=start_date, end=end_date)

        # Format price data
        price_data = price_data.round(2)
        
        return asset_name, price_data

    def prepare_asset_for_db(self, ticker: str, asset_name: str):
        """
        Prepares asset data for database insertion.
        
        :param ticker: The stock ticker symbol
        :param asset_name: The name of the asset
        :return: Dictionary with asset data
        """
        return {
            'ticker': ticker,
            'name': asset_name
        }

    def prepare_price_history_for_db(self, price_data: pd.DataFrame):
        """
        Prepares price history data for database insertion.
        
        :param ticker: The stock ticker symbol
        :param price_data: DataFrame with price history
        :return: List of dictionaries with price history data
        """
        price_history = price_data.reset_index()
        price_history.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        price_history = price_history.to_dict('records')

        return price_history

# Usage example
if __name__ == "__main__":
    

    from database.access import DatabaseAccess
    db_access = DatabaseAccess()
    db_access.init_db()

    yfhandler = YFinanceService()

    ticker = 'AAPL'
    asset_name, price_data = yfhandler.fetch_asset_data(ticker)
    
    # Prepare and insert asset data
    asset_data = yfhandler.prepare_asset_for_db(ticker, asset_name)
    db_access.add_asset(**asset_data)
    
    # Prepare and insert price history data
    price_history = yfhandler.prepare_price_history_for_db(price_data)
    db_access.add_price_history(ticker, price_history)

    print(f"Added data for {ticker}")