
import logging
import pandas as pd

from database.access import DatabaseAccess


class MetricService:

    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access

    def get_holdings_in_base_currency(self, base_currency_id: int, portfolio_id: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:

        # Fetch raw data
        holdings = self.db_access.get_portfolio_holdings_time_series(portfolio_id, start_date, end_date)
        asset_prices = self.db_access.get_asset_prices(start_date, end_date)
        base_currency_rates = self.db_access.get_currency_rates(base_currency_id, start_date, end_date)

        # Convert to pandas DataFrames
        holdings_df = pd.DataFrame(holdings)
        asset_prices_df = pd.DataFrame(asset_prices)
        base_currency_rates_df = pd.DataFrame(base_currency_rates)

        # Merge and calculate
        result = holdings_df.merge(asset_prices_df, on=['date', 'asset_id'])
        result['value_usd'] = result['quantity'] * result['price']
        result = result.merge(base_currency_rates_df, on='date')
        result['value_base_currency'] = result['value_usd'] / result['exchange_rate']

        return result

        return

    