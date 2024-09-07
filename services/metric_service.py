
import logging
import pandas as pd

from database.access import DatabaseAccess, with_session, Session

class MetricService:

    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access

    @with_session
    def get_holdings_in_base_currency(self, session: Session, portfolio_id: int, asset_id: int, currency_id: int) -> pd.DataFrame:
        """
        Calculate holdings in base currency for a given portfolio and date range.

        This method fetches data from the database and performs all calculations in memory
        using pandas DataFrames. No intermediate results are stored back to the database.

        Args:
            currency_id (int): The ID of the base currency to convert holdings to.
            portfolio_id (int): The ID of the portfolio to calculate holdings for.
            start_date (datetime): The start date of the date range to consider.
            end_date (datetime): The end date of the date range to consider.

        Returns:
            pd.DataFrame: A DataFrame containing the holdings in base currency.
        """

        # Fetch raw data (data returned directly to DataFrame)
        in_kind_ts = self.db_access.get_portfolio_asset_time_series_df(session, portfolio_id, asset_id)

        if len(in_kind_ts) > 0:
            # Assuming in_kind is a list of dictionaries with a 'date' or 'timestamp' field
            start_date = in_kind_ts.iloc[0]['date']  # or 'timestamp', depending on your data structure
            end_date = in_kind_ts.iloc[-1]['date']  # or 'timestamp'
        else:
            return
        
        # Fetch asset and currency price history (data returned directly to DataFrames)
        asset_prices = self.db_access.get_asset_price_history_df(session, asset_id, start_date, end_date)
        asset_prices.rename(columns={'close': 'close_asset'}, inplace=True)
        currency_prices = self.db_access.get_asset_price_history_df(session, currency_id, start_date, end_date)
        currency_prices.rename(columns={'close': 'close_currency'}, inplace=True)

        # Merge the three DataFrames on date
        merged_df = pd.merge(in_kind_ts, asset_prices, on='date', how='outer', suffixes=('', '_asset'))
        merged_df = pd.merge(merged_df, currency_prices, on='date', how='outer', suffixes=('', '_currency'))

        # Ffill the dataframe (assume weekend values equal to close on friday)
        merged_df = merged_df.ffill()

        # Calculate the value in base currency
        merged_df['value_in_currency'] = merged_df['quantity'] * merged_df['close_asset'] / merged_df['close_currency']

        return merged_df


if __name__ == "__main__":

        metric_service = MetricService(DatabaseAccess())
        result = metric_service.get_holdings_in_base_currency(1, 4, 2)  # Example portfolio, asset, and base currency IDs

        print('done')