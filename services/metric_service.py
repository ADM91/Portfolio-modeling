
import logging
import pandas as pd
from datetime import datetime

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
        df['value'] = 0.0

        # Convert value invested to specified currency
        for i, row in df.iterrows():
            sign = 1 if row['action_type_name'] == 'buy' else -1
            if currency_id != row['currency_id']:
                conversion_rate = self.db_access.get_currency_conversion_on_date(session, row['currency_id'], currency_id, row['date'])
                df.at[i, 'value'] = sign * conversion_rate * row['price'] * row['quantity']
            else:
                df.at[i, 'value'] = sign * row['price'] * row['quantity']
        
        # Create a daily date range
        date_range = pd.date_range(start=df['date'].min(), end=datetime.now(), freq='D')
        
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

    asset = 4
    currency = 2


    metric_service = MetricService(DatabaseAccess())
    result = metric_service.get_holdings_in_base_currency(portfolio_id=1, asset_id=asset, currency_id=currency)  # Example portfolio, asset, and base currency IDs        
    result3 = metric_service.get_value_invested(portfolio_id=1, asset_id=asset, currency_id=currency)  # Example portfolio, asset, and base currency IDs

    print('done')

    import matplotlib.pyplot as plt
    import numpy as np

    # Convert to DataFrames
    df1 = pd.DataFrame(result)
    df3 = pd.DataFrame(result3)

    # Merge the dataframes on date
    merged_df = pd.merge(df1, df3, on='date', how='outer').sort_values('date')
    
    # Forward fill any missing values
    merged_df = merged_df.ffill()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert values to millions of ISK
    holdings = merged_df['value_in_currency']
    invested = merged_df['cumulative_value']
    dates = merged_df['date']

    # Plot holdings in base currency
    ax.plot(dates, holdings, color='blue', label='Holdings in Base Currency')

    # Create shaded area for value invested
    ax.fill_between(dates, 0, invested, 
                    where=(invested >= 0), color='red', alpha=0.3, 
                    label='Positive Value Invested')
    ax.fill_between(dates, 0, invested, 
                    where=(invested < 0), color='green', alpha=0.3, 
                    label='Negative Value Invested')

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('currency')
    ax.set_title('Holdings in Base Currency and Value Invested Over Time')

    # Add legend
    ax.legend(loc='upper left')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('holdings_and_invested_value.png')
    print('Plot saved as holdings_and_invested_value.png')

