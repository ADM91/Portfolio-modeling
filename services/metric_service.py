
import logging
import pandas as pd
from datetime import datetime
from typing import List

from database.access import DatabaseAccess, with_session, Session


class MetricService:

    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access

    def _get_holdings_in_base_currency(self, session: Session, portfolio_id: int, asset_id: int, currency_id: int) -> pd.DataFrame:
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

        in_kind_ts = self.db_access.get_portfolio_asset_time_series_df(session, portfolio_id, asset_id)

        if not in_kind_ts.empty:
            start_date, end_date = in_kind_ts['date'].min(), in_kind_ts['date'].max()

            asset_prices = self.db_access.get_asset_price_history_df(session, asset_id, start_date, end_date)
            currency_prices = self.db_access.get_asset_price_history_df(session, currency_id, start_date, end_date)

            merged_df = pd.merge(in_kind_ts, asset_prices[['date', 'close']], on='date', how='outer')
            merged_df = pd.merge(merged_df, currency_prices[['date', 'close']], on='date', how='outer', suffixes=('_asset', '_currency'))

            merged_df['value_in_currency'] = merged_df['quantity'] * merged_df['close_asset'] / merged_df['close_currency']
            merged_df['portfolio_id'] = portfolio_id
            merged_df['asset_id'] = asset_id

            return merged_df[['date', 'portfolio_id', 'asset_id', 'value_in_currency']]
        else:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'value_in_currency'])

    @with_session
    def get_holdings_in_base_currency_general(self, session: Session, portfolio_ids: List[int], asset_ids: List[int], currency_id: int) -> pd.DataFrame:
        """
        Calculate holdings in base currency for given portfolios and assets over their entire date range.

        This method fetches data from the database and performs all calculations in memory
        using pandas DataFrames. No intermediate results are stored back to the database.

        Args:
            session (Session): The database session.
            portfolio_ids (List[int]): The IDs of the portfolios to calculate holdings for.
            asset_ids (List[int]): The IDs of the assets to calculate holdings for.
            currency_id (int): The ID of the base currency to convert holdings to.

        Returns:
            pd.DataFrame: A melted DataFrame containing the holdings in base currency.
        """
        all_data = []

        for portfolio_id in portfolio_ids:
            for asset_id in asset_ids:
                holdings_df = self._get_holdings_in_base_currency(session, portfolio_id, asset_id, currency_id)
                if not holdings_df.empty:
                    all_data.append(holdings_df)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df = final_df.sort_values(['date', 'portfolio_id', 'asset_id']).reset_index(drop=True)
            final_df.rename(columns={'value_in_currency': 'value_holdings'}, inplace=True)
            
            return final_df
        else:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'value_holdings'])

    def _get_value_invested(self, session: Session, portfolio_id: int, asset_id: int, currency_id: int) -> pd.DataFrame:
        """
        Calculate the value invested in a specific asset for a given portfolio.
        
        Args:
            portfolio_id (int): The ID of the portfolio.
            asset_id (int): The ID of the asset.
            currency_id (int): The ID of the currency to convert the value to.

        Returns:
            pd.DataFrame: A DataFrame containing the cumulative value invested over time.
        """
        actions = self.db_access.get_buy_sell_actions_by_portfolio_id_asset_id(session, portfolio_id, asset_id)
        
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

        # Reset the index to make 'date' a column
        daily_df = daily_df.reset_index()[['date', 'value', 'cumulative_value']]
        
        return daily_df

    @with_session
    def get_value_invested_general(self, session: Session, portfolio_ids: List[int], asset_ids: List[int], currency_id: int) -> pd.DataFrame:
        """
        Calculate the value invested for multiple portfolios and assets.
        
        Args:
            session (Session): The database session.
            portfolio_ids (List[int]): The IDs of the portfolios to calculate value invested for.
            asset_ids (List[int]): The IDs of the assets to calculate value invested for.
            currency_id (int): The ID of the currency to convert the value to.

        Returns:
            pd.DataFrame: A melted DataFrame containing the cumulative value invested over time for all portfolio-asset combinations.
        """
        all_data = []

        for portfolio_id in portfolio_ids:
            for asset_id in asset_ids:
                value_invested_df = self._get_value_invested(session, portfolio_id, asset_id, currency_id)
                value_invested_df['portfolio_id'] = portfolio_id
                value_invested_df['asset_id'] = asset_id
                all_data.append(value_invested_df)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df = final_df.sort_values(['date', 'portfolio_id', 'asset_id']).reset_index(drop=True)
            final_df.rename(columns={'value':'value_invested_discrete', 'cumulative_value': 'value_invested_cumulative'}, inplace=True)
            
            return final_df
        else:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'value_invested_discrete', 'value_invested_cumulative'])

    @with_session
    def get_cost_basis(self, session: Session, portfolio_id: int, asset_id: int, currency_id: int) -> pd.DataFrame:
        """
        Calculate the cost basis of a specific asset for a given portfolio over time.
        
        Args:
            session (Session): The database session.
            portfolio_id (int): The ID of the portfolio.
            asset_id (int): The ID of the asset.
            currency_id (int): The ID of the currency to calculate the cost basis in.

        Returns:
            pd.DataFrame: A DataFrame containing the cost basis over time.
        """
        actions = self.db_access.get_buy_sell_actions_by_portfolio_id_asset_id(session, portfolio_id, asset_id)
        
        df = pd.read_sql(actions.statement, session.bind)
        df = df.sort_values(by='date').reset_index(drop=True)

        df['quantity'] = df.apply(lambda row: row['quantity'] if row['action_type_name'] == 'buy' else -row['quantity'], axis=1)
        df['cost'] = 0.0
        df['cumulative_quantity'] = 0.0
        df['cumulative_cost'] = 0.0

        for i, row in df.iterrows():
            if row['action_type_name'] == 'buy':
                if currency_id != row['currency_id']:
                    conversion_rate = self.db_access.get_currency_conversion_on_date(session, row['currency_id'], currency_id, row['date'])
                    df.at[i, 'cost'] = conversion_rate * row['price'] * row['quantity']
                else:
                    df.at[i, 'cost'] = row['price'] * row['quantity']
            else:  # sell action
                if df.at[i-1, 'cumulative_quantity'] > 0:
                    df.at[i, 'cost'] = row['quantity'] * df.at[i-1, 'cost_basis']  # sale cost = quantity * cost basis

            
            df.at[i, 'cumulative_quantity'] = df['quantity'][:i+1].sum()
            df.at[i, 'cumulative_cost'] = df['cost'][:i+1].sum()
            
            if df.at[i, 'cumulative_quantity'] > 0:
                df.at[i, 'cost_basis'] = df.at[i, 'cumulative_cost'] / df.at[i, 'cumulative_quantity']
            else:
                df.at[i, 'cost_basis'] = 0  # Reset cost basis when all assets are sold

        # Create a daily date range
        date_range = pd.date_range(start=df['date'].min(), end=datetime.now(), freq='D')
        
        # Create a daily DataFrame
        daily_df = pd.DataFrame(index=date_range)
        daily_df.index.name = 'date'
        
        # Merge the action data with the daily DataFrame
        daily_df = daily_df.merge(df[['date', 'cumulative_quantity', 'cumulative_cost']], left_index=True, right_on='date', how='left')
        
        # Fill forward the values
        daily_df['cumulative_quantity'] = daily_df['cumulative_quantity'].ffill()
        daily_df['cumulative_cost'] = daily_df['cumulative_cost'].ffill()
        
        # Calculate the cost basis
        daily_df['cost_basis'] = daily_df['cumulative_cost'] / daily_df['cumulative_quantity']
        
        # Reset the index to make 'date' a column
        daily_df = daily_df.reset_index()[['date', 'cumulative_quantity', 'cumulative_cost', 'cost_basis']]
        
        return daily_df

    @with_session
    def get_unrealized_gain_loss(self, session: Session, portfolio_id: int, asset_id: int, currency_id: int) -> pd.DataFrame:
        """
        Calculate the unrealized gain/loss for a specific asset in a given portfolio.

        Args:
            session (Session): The database session.
            portfolio_id (int): The ID of the portfolio.
            asset_id (int): The ID of the asset.
            currency_id (int): The ID of the currency to calculate the gain/loss in.

        Returns:
            pd.DataFrame: A DataFrame containing the unrealized gain/loss over time.
        """
        # Get cost basis
        cost_basis_df = self.get_cost_basis(portfolio_id, asset_id, currency_id)

        # Get holdings in base currency
        holdings_df = self._get_holdings_in_base_currency(session, portfolio_id, asset_id, currency_id)

        # Merge cost basis and holdings dataframes
        merged_df = pd.merge(cost_basis_df, holdings_df, on='date', how='outer')
        merged_df = merged_df.sort_values('date').ffill()

        # Calculate unrealized gain/loss
        merged_df['unrealized_gain_loss'] = merged_df['value_in_currency'] - merged_df['cumulative_cost']

        # Calculate unrealized gain/loss percentage
        merged_df['unrealized_gain_loss_percentage'] = (merged_df['unrealized_gain_loss'] / merged_df['cumulative_cost']) * 100

        return merged_df[['date', 'cumulative_quantity', 'cumulative_cost', 'cost_basis', 'value_in_currency', 'unrealized_gain_loss', 'unrealized_gain_loss_percentage']]

    @with_session
    def get_unrealized_gain_loss_general(self, session: Session, portfolio_ids: List[int], asset_ids: List[int], currency_id: int) -> pd.DataFrame:
        """
        Calculate the unrealized gain/loss for multiple portfolios and assets.

        Args:
            session (Session): The database session.
            portfolio_ids (List[int]): The IDs of the portfolios to calculate unrealized gain/loss for.
            asset_ids (List[int]): The IDs of the assets to calculate unrealized gain/loss for.
            currency_id (int): The ID of the currency to calculate the gain/loss in.

        Returns:
            pd.DataFrame: A melted DataFrame containing the unrealized gain/loss over time for all portfolio-asset combinations.
        """
        all_data = []

        for portfolio_id in portfolio_ids:
            for asset_id in asset_ids:
                unrealized_gain_loss_df = self.get_unrealized_gain_loss(portfolio_id, asset_id, currency_id)
                unrealized_gain_loss_df['portfolio_id'] = portfolio_id
                unrealized_gain_loss_df['asset_id'] = asset_id
                all_data.append(unrealized_gain_loss_df)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df = final_df.sort_values(['date', 'portfolio_id', 'asset_id']).reset_index(drop=True)
            
            return final_df
        else:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'cumulative_quantity', 'cumulative_cost', 'cost_basis', 'value_in_currency', 'unrealized_gain_loss', 'unrealized_gain_loss_percentage'])

    @with_session
    def get_time_weighted_return_general(self, session: Session, portfolio_ids: List[int], asset_ids: List[int], currency_id: int) -> pd.DataFrame:

        # Get the unrealized gain/loss data for all portfolio-asset combinations
        total_unrealized_gain_loss = self.get_unrealized_gain_loss_general(portfolio_ids=[1,2], asset_ids=[4,5,6], currency_id=1)

        # aggregate on date
        tugl_agg = self.aggregate_on_date(df=total_unrealized_gain_loss, column='unrealized_gain_loss')

        return

    def aggregate_on_date(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:

        # Aggregate the dataframe on 'date' and sum the specified column
        aggregated_df = df.groupby('date')[columns].sum().reset_index()

        return aggregated_df





if __name__ == "__main__":

    import time
    from frontend.graphs import multi_portfolio_value_graph, holdings_and_invested_value_graph_v2, holdings_and_cost_basis_graph, price_and_cost_basis, unrealized_gain_loss_percentage_graph, total_unrealized_gain_loss_graph

    # TODO: DONE ~~debug spike in holdings_and_invested_value graph~~
    # TODO: DONE ~~Aggregate portfolio-asset view on holdings_in_base_currency and value_invested~~
    # TODO: DONE ~~cost basis and unrealized gain/loss~~
    # TODO: realized gain/loss, each sale 1 line - fields: realized gain/loss, implied tax
    # TODO: time-weighted return
    # TODO: benchmark comparison - time-weighted return and total return (based on value_invested)
    # TODO: sharpe ratio - on what? (time-weighted return - like a piecewise return - break down return into periods of constant value invested)

    metric_service = MetricService(DatabaseAccess())

    start_time = time.time()
    result_holdings_value = metric_service.get_holdings_in_base_currency_general(portfolio_ids=[1,2], asset_ids=[4,5,6], currency_id=1)  # Example portfolio, asset, and base currency IDs        
    end_time = time.time()
    print(f"Runtime for get_holdings_in_base_currency: {end_time - start_time:.3f} seconds")

    start_time = time.time()
    result_invested_value = metric_service.get_value_invested_general(portfolio_ids=[1,2], asset_ids=[4,5,6], currency_id=1)  # Example portfolio, asset, and base currency IDs
    end_time = time.time()
    print(f"Runtime for get_value_invested: {end_time - start_time:.3f} seconds")

    start_time = time.time()
    result_cost_basis = metric_service.get_cost_basis(portfolio_id=1, asset_id=4, currency_id=1)  # Example portfolio, asset, and base currency IDs
    end_time = time.time()
    print(f"Runtime for get_cost_basis: {end_time - start_time:.3f} seconds")

    start_time = time.time()
    result_unrealized_gain_loss = metric_service.get_unrealized_gain_loss(portfolio_id=1, asset_id=4, currency_id=1)  # Example portfolio, asset, and base currency IDs
    end_time = time.time()
    print(f"Runtime for get_unrealized_gain_loss: {end_time - start_time:.3f} seconds")

    start_time = time.time()
    result_total_unrealized_gain_loss = metric_service.get_unrealized_gain_loss_general(portfolio_ids=[1,2], asset_ids=[4,5,6], currency_id=1)  # Example portfolio, asset, and base currency IDs
    end_time = time.time()
    print(f"Runtime for get_unrealized_gain_loss: {end_time - start_time:.3f} seconds")

    # graph
    multi_portfolio_value_graph(result_holdings_value)
    holdings_and_invested_value_graph_v2(result_holdings_value, result_invested_value)
    holdings_and_cost_basis_graph(result_holdings_value, result_cost_basis, portfolio_id=1, asset_id=4, currency_id=1)
    price_and_cost_basis(result_holdings_value, result_cost_basis, portfolio_id=1, asset_id=4, currency_id=1)
    unrealized_gain_loss_percentage_graph(result_unrealized_gain_loss, portfolio_id=1, asset_id=4, currency_id=1)
    total_unrealized_gain_loss_graph(result_total_unrealized_gain_loss)

    # @with_session
    # def get_currency_conversion_rates(self, session: Session, from_currency_id: int, to_currency_id: int, start_date, end_date) -> pd.DataFrame:
    #     """
    #     Get currency conversion rates for a given date range.

    #     Args:
    #         session (Session): The database session.
    #         from_currency_id (int): The ID of the currency to convert from.
    #         to_currency_id (int): The ID of the currency to convert to.
    #         start_date (datetime): The start date of the conversion range.
    #         end_date (datetime): The end date of the conversion range.

    #     Returns:
    #         pd.DataFrame: A DataFrame containing daily conversion rates.
    #     """

    #     conversion_rates = self.db_access.get_currency_conversion_time_series(session, from_currency_id, to_currency_id, start_date, end_date)

    #     # Convert to DataFrame
    #     df = pd.read_sql(conversion_rates.statement, session.bind)

    #     # Ensure the date column is datetime type
    #     df['date'] = pd.to_datetime(df['date'])
        
    #     # Calculate conversion rate
    #     df['conversion_rate'] = df['from_currency_price'] / df['to_currency_price']

    #     # Create a complete date range
    #     date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    #     full_df = pd.DataFrame(index=date_range)
    #     full_df.index.name = 'date'

    #     # Merge with the conversion rates and forward fill missing values
    #     result_df = full_df.merge(df[['date', 'conversion_rate']], left_index=True, right_on='date', how='left')
    #     result_df['conversion_rate'] = result_df['conversion_rate'].ffill()
    #     result_df['conversion_rate'] = result_df['conversion_rate'].bfill()
    #     result_df = result_df.reset_index(drop=True)

    #     return result_df
