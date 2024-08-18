
import logging
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

from database.access_v2 import with_session, get_all_assets, get_last_price_date, add_price_history


class YFinanceService:
    """
    Class for fetching asset data from yFinance API and storing it in the database.
    """
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
        
        # Fetch price history data
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
        price_history['date'] = price_history['date'].apply(lambda x: x.date())
        price_history = price_history.to_dict('records')

        return price_history

    @with_session
    def update_db_with_asset_data(self, session):
        asset_list = get_all_assets(session)

        # omit USD - we assume always USD = 1
        asset_list = [asset for asset in asset_list if asset.code != 'USD']
        
        for asset in asset_list:
            # Get the most recent date in the database for this asset
            last_date = get_last_price_date(session, asset.ticker)
            
            # If we have no data, start from 5 years ago, otherwise start from the day after the last date
            start_date = last_date if last_date else datetime.now() - timedelta(days=5*365)
            end_date = datetime.now()

            # Only fetch data if there's a gap to fill
            if start_date < end_date:
                logging.info(f"Updating {asset.ticker} from {start_date} to {end_date}")
                
                try:
                    # Fetch asset data - close on current day represents current price
                    asset_name, price_data = self.fetch_asset_data(asset.ticker, start_date, end_date)
                    
                    # uninvert if inverted
                    if asset.is_inverted:
                        price_data[['Open', 'High', 'Low', 'Close']] = price_data[['Open', 'High', 'Low', 'Close']].apply(lambda x: round(1/x, 6))

                    if not price_data.empty:
                        # Prepare and insert price history data
                        price_history = self.prepare_price_history_for_db(price_data)
                        add_price_history(session, asset.ticker, price_history)
                        logging.info(f"Updated {asset.ticker} with {len(price_history)} new entries")
                    else:
                        logging.info(f"No new data for {asset.ticker}")
                except Exception as e:
                    logging.error(f"Error updating {asset.ticker}: {str(e)}")
            else:
                logging.info(f"{asset.ticker} is up to date")


# Usage example
if __name__ == "__main__":
    
    yf_service = YFinanceService()
    yf_service.update_db_with_asset_data()

    print('done')