
import logging
import pandas as pd

from database.entities import Action, Asset, PriceHistory
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
    
    @with_session
    def get_value_invested(self, session: Session, portfolio_id: int, asset_id: int, currency_id: int) -> pd.DataFrame:
        """
        Calculate the value invested in a specific asset for a given portfolio.
        
        Args:
            portfolio_id (int): The ID of the portfolio.
            asset_id (int): The ID of the asset.
            currency_id (int): The ID of the currency to convert the value to.

        Returns:
            pd.DataFrame: A DataFrame containing the cumulative value invested over time.
        """
        actions = self.db_access.get_actions_by_portfolio_id_asset_id(session, portfolio_id, asset_id)
        
        df = pd.read_sql(actions.statement, session.bind)
        df = df.sort_values(by='date').reset_index(drop=True)

        # calcuate value of buy/sell in currency
        df['value'] = df['price'] * df['quantity']

        # Convert to specified currency if necessary
        if currency_id != df['currency_id'].iloc[0]:
            conversion_rates = self.get_currency_conversion_rates(session, df['currency_id'].iloc[0], currency_id, df['date'].min(), df['date'].max())
            df = pd.merge(df, conversion_rates, on='date', how='left')
            df['value'] = df['value'] * df['conversion_rate']
        
        # Create a daily date range
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
        
        # Create a daily DataFrame
        daily_df = pd.DataFrame(index=date_range)
        daily_df.index.name = 'date'
        
        # Merge the action data with the daily DataFrame
        daily_df = daily_df.merge(df[['date', 'value']], left_index=True, right_on='date', how='left')
        
        # Fill forward the values (this assumes that the value remains constant until the next transaction)
        daily_df['cumulative_value'] = daily_df['value'].fillna(0).cumsum().ffill()
        
        return daily_df.reset_index()[['date', 'cumulative_value']]

    @with_session
    def get_currency_conversion_rates(self, session: Session, from_currency_id: int, to_currency_id: int, start_date, end_date) -> pd.DataFrame:
        """
        Get currency conversion rates for a given date range.

        Args:
            session (Session): The database session.
            from_currency_id (int): The ID of the currency to convert from.
            to_currency_id (int): The ID of the currency to convert to.
            start_date (datetime): The start date of the conversion range.
            end_date (datetime): The end date of the conversion range.

        Returns:
            pd.DataFrame: A DataFrame containing daily conversion rates.
        """

        conversion_rates = self.db_access.get_currency_conversion_time_series(session, from_currency_id, to_currency_id, start_date, end_date)

        # Convert to DataFrame
        df = pd.read_sql(conversion_rates.statement, session.bind)

        # Ensure the date column is datetime type
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate conversion rate
        df['conversion_rate'] = df['from_currency_price'] / df['to_currency_price']

        # Create a complete date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        full_df = pd.DataFrame(index=date_range)
        full_df.index.name = 'date'

        # Merge with the conversion rates and forward fill missing values
        result_df = full_df.merge(df[['date', 'conversion_rate']], left_index=True, right_on='date', how='left')
        result_df['conversion_rate'] = result_df['conversion_rate'].ffill()
        result_df['conversion_rate'] = result_df['conversion_rate'].bfill()
        result_df = result_df.reset_index(drop=True)

        return result_df


if __name__ == "__main__":

        metric_service = MetricService(DatabaseAccess())
        result = metric_service.get_holdings_in_base_currency(1, 4, 2)  # Example portfolio, asset, and base currency IDs

        result2 = metric_service.get_currency_conversion_rates(from_currency_id=2, to_currency_id=3, start_date='2023-01-01', end_date='2023-12-31')  # Example from and to currency IDs, and date range
        
        result3 = metric_service.get_value_invested(portfolio_id=1, asset_id=4, currency_id=2)  # Example portfolio, asset, and base currency IDs

        print('done')

        from matplotlib import pyplot as plt

        plt.plot(result['date'], result['value_in_currency'])
        plt.savefig('my_plot.png')
        print('done')