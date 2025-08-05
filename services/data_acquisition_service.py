import logging
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session
import time
import random
from typing import Optional

from database.access import with_session, DatabaseAccess
from services.data_providers import DataProvider, YFinanceProvider


class DataAcquisitionService:
    """
    Generalized service for fetching asset data from various providers and storing it in the database.
    """

    def __init__(self, db_access: DatabaseAccess, data_provider: Optional[DataProvider] = None):
        self.db_access = db_access
        self.data_provider = data_provider or YFinanceProvider()  # Default to Yahoo Finance

    def fetch_asset_data(self, ticker: str, start_date: datetime = None, end_date: datetime = None):
        """
        Fetches asset data using the configured data provider.
        
        :param ticker: The stock ticker symbol
        :param start_date: Start date for historical data (default: 5 years ago)
        :param end_date: End date for historical data (default: today)
        :return: Tuple containing asset info and price history
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=5*365)  # 5 years ago
        if not end_date:
            end_date = datetime.now()

        return self.data_provider.fetch_asset_data(ticker, start_date, end_date)

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
        Standardizes column names regardless of the data provider.
        
        :param price_data: DataFrame with price history
        :return: List of dictionaries with price history data
        """
        price_history = price_data.reset_index()
        
        # Standardize column names (handle different provider formats)
        column_mapping = {
            'Date': 'date',
            'Open': 'open', 
            'High': 'high', 
            'Low': 'low', 
            'Close': 'close', 
            'Volume': 'volume'
        }
        
        price_history.rename(columns=column_mapping, inplace=True)
        
        # Convert date to date object if it's a datetime
        if 'date' in price_history.columns:
            price_history['date'] = price_history['date'].apply(
                lambda x: x.date() if hasattr(x, 'date') else x
            )
        
        return price_history.to_dict('records')

    def _create_usd_price_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Creates price data for USD (always 1.0).
        
        :param start_date: Start date
        :param end_date: End date
        :return: DataFrame with USD price data
        """
        date_range = pd.date_range(start=start_date, end=end_date)
        return pd.DataFrame({
            'Date': date_range,
            'Open': 1.0,
            'High': 1.0,
            'Low': 1.0,
            'Close': 1.0,
            'Volume': 0
        })

    def _apply_inversion(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies inversion to price data (1/x) for inverted assets.
        
        :param price_data: Original price data
        :return: Inverted price data
        """
        price_columns = ['Open', 'High', 'Low', 'Close']
        price_data[price_columns] = price_data[price_columns].apply(
            lambda x: round(1/x, 6)
        )
        return price_data

    @with_session
    def update_db_with_asset_data(self, session: Session):
        """
        Updates the database with latest asset data for all assets.
        """
        asset_list = self.db_access.get_all_assets(session)
        
        logging.info(f"Starting data update using {self.data_provider.get_provider_name()}")
        
        for asset in asset_list:
            # Add a small delay between requests to be respectful to APIs
            time.sleep(random.uniform(0.2, 0.8))

            # Get the most recent date in the database for this asset
            last_date = self.db_access.get_last_price_date(session, asset.ticker)
            
            # If we have no data, start from 5 years ago, otherwise start from the day after the last date
            start_date = last_date + timedelta(days=1) if last_date else datetime.now() - timedelta(days=5*365)
            end_date = datetime.now()

            # Only update if there's a gap to fill
            if start_date <= end_date:
                logging.info(f"Updating {asset.ticker} from {start_date} to {end_date}")
                
                try:
                    if asset.code == 'USD':
                        # For USD, create a DataFrame with all values set to 1
                        price_data = self._create_usd_price_data(start_date, end_date)
                    else:
                        # Fetch asset data for non-USD assets
                        asset_name, price_data = self.fetch_asset_data(asset.ticker, start_date, end_date)
                        
                        # Apply inversion if needed
                        if asset.is_inverted:
                            price_data = self._apply_inversion(price_data)

                    if not price_data.empty:
                        # Prepare and insert price history data
                        price_history = self.prepare_price_history_for_db(price_data)
                        self.db_access.add_price_history(session, asset.ticker, price_history)
                        logging.info(f"Updated {asset.ticker} with {len(price_history)} new entries")
                    else:
                        logging.info(f"No new data for {asset.ticker}")
                        
                except Exception as e:
                    logging.error(f"Error updating {asset.ticker} using {self.data_provider.get_provider_name()}: {str(e)}")
            else:
                logging.info(f"{asset.ticker} is up to date")

    def switch_provider(self, new_provider: DataProvider):
        """
        Switch to a different data provider.
        
        :param new_provider: The new data provider to use
        """
        old_provider = self.data_provider.get_provider_name()
        self.data_provider = new_provider
        logging.info(f"Switched data provider from {old_provider} to {new_provider.get_provider_name()}")


# Usage example
if __name__ == "__main__":
    from database.access import DatabaseAccess
    from services.data_providers import YFinanceProvider

    # Using default provider (Yahoo Finance)
    service = DataAcquisitionService(DatabaseAccess())
    service.update_db_with_asset_data()

    # Or explicitly specify a provider
    # yf_provider = YFinanceProvider()
    # service = DataAcquisitionService(DatabaseAccess(), yf_provider)
    # service.update_db_with_asset_data()

    print('Data acquisition complete')
