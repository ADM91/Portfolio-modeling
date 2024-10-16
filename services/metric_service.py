
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import List

from database.access import DatabaseAccess, with_session, Session


class MetricService:

    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access

    @with_session
    def _get_holdings_in_base_currency(self, session: Session, portfolio_id: int, asset_id: int, currency_id: int, start_date: date=date(1970, 1, 1), end_date: date=datetime.now().date()) -> pd.DataFrame:
        """
        Calculate holdings in base currency for a given portfolio and date range.

        Args:
            session (Session): The database session.
            portfolio_id (int): The ID of the portfolio to calculate holdings for.
            asset_id (int): The ID of the asset to calculate holdings for.
            currency_id (int): The ID of the base currency to convert holdings to.
            start_date (date): The start date of the date range to consider.
            end_date (date): The end date of the date range to consider.

        Returns:
            pd.DataFrame: A DataFrame containing the holdings in base currency.
        """

        in_kind_ts = self.db_access.get_portfolio_asset_time_series_df(session, portfolio_id, asset_id)

        if not in_kind_ts.empty:
            # start_date, end_date = in_kind_ts['date'].min(), in_kind_ts['date'].max()
            in_kind_ts = in_kind_ts[(in_kind_ts['date'] >= pd.to_datetime(start_date)) & (in_kind_ts['date'] <= pd.to_datetime(end_date))]

            asset_prices = self.db_access.get_asset_price_history_df(session, asset_id, start_date, end_date)
            currency_prices = self.db_access.get_asset_price_history_df(session, currency_id, start_date, end_date)

            df = pd.merge(in_kind_ts, asset_prices[['date', 'close']], on='date', how='left')
            df = pd.merge(df, currency_prices[['date', 'close']], on='date', how='left', suffixes=('_asset', '_currency'))

            df['value_in_currency'] = df['quantity'] * df['close_asset'] / df['close_currency']
            df['portfolio_id'] = portfolio_id
            df['asset_id'] = asset_id

            return df[['date', 'portfolio_id', 'asset_id', 'value_in_currency']]
        else:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'value_in_currency'])
    
    def get_holdings_in_base_currency_general(self, portfolio_ids: List[int], asset_ids: List[int], currency_id: int, start_date: date=date(1970, 1, 1), end_date: date=datetime.now().date()) -> pd.DataFrame:
        """
        Calculate holdings in base currency for given portfolios and assets over a specified date range.

        Args:
            session (Session): The database session.
            portfolio_ids (List[int]): The IDs of the portfolios to calculate holdings for.
            asset_ids (List[int]): The IDs of the assets to calculate holdings for.
            currency_id (int): The ID of the base currency to convert holdings to.
            start_date (date): The start date of the date range to consider.
            end_date (date): The end date of the date range to consider.

        Returns:
            pd.DataFrame: A melted DataFrame containing the holdings in base currency.
        """
        all_data = []

        for portfolio_id in portfolio_ids:
            for asset_id in asset_ids:
                holdings_df = self._get_holdings_in_base_currency(portfolio_id, asset_id, currency_id, start_date, end_date)
                if not holdings_df.empty:
                    all_data.append(holdings_df)

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values(['date', 'portfolio_id', 'asset_id']).reset_index(drop=True)
            df.rename(columns={'value_in_currency': 'value_holdings'}, inplace=True)
            
            return df[['date', 'portfolio_id', 'asset_id', 'value_holdings']]
        else:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'value_holdings'])

    @with_session
    def _get_cash_flow(self, session: Session, portfolio_id: int, asset_id: int, currency_id: int, start_date: date = date(1970, 1, 1), end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Calculate the value invested in a specific asset for a given portfolio over a specified date range.
        
        Args:
            session (Session): The database session.
            portfolio_id (int): The ID of the portfolio.
            asset_id (int): The ID of the asset.
            currency_id (int): The ID of the currency to convert the value to.
            start_date (date): The start date of the date range to consider.
            end_date (date): The end date of the date range to consider.

        Returns:
            pd.DataFrame: A DataFrame containing the cumulative value invested over time.
        """
        actions = self.db_access.get_buy_sell_actions_by_portfolio_id_asset_id(session, portfolio_id, asset_id)
        
        actions_df = pd.read_sql(actions.statement, session.bind)
        actions_df = actions_df.sort_values(by='date').reset_index(drop=True)

        # calcuate value of buy/sell in currency
        actions_df['cash_flow'] = 0.0

        # Convert value invested to specified currency
        for i, row in actions_df.iterrows():
            sign = 1 if row['action_type_name'] == 'buy' else -1
            if currency_id != row['currency_id']:
                conversion_rate = self.db_access.get_currency_conversion_on_date(session, row['currency_id'], currency_id, row['date'])
                actions_df.at[i, 'cash_flow'] = sign * conversion_rate * row['price'] * row['quantity']
            else:
                actions_df.at[i, 'cash_flow'] = sign * row['price'] * row['quantity']
        
        # Aggregate daily cash flow
        actions_df = actions_df.groupby(['date', 'portfolio_id', 'asset_id'])['cash_flow'].sum().reset_index()
        actions_df.rename(columns={'cash_flow': 'cash_flow_daily'}, inplace=True)

        # Create a daily date range
        date_range = pd.date_range(start=max(start_date, actions_df['date'].min().date()), end=min(end_date, datetime.now().date()), freq='D')
        
        # Create a daily DataFrame
        df = pd.DataFrame(index=date_range)
        df.index.name = 'date'
        
        # Merge the action data with the daily DataFrame
        df = df.merge(actions_df[['date', 'portfolio_id', 'asset_id', 'cash_flow_daily']], left_index=True, right_on='date', how='left')
        
        # Fill forward the values (this assumes that the value remains constant until the next transaction)
        df['cash_flow_cumulative'] = df['cash_flow_daily'].fillna(0).cumsum().ffill()
        df['portfolio_id'] = portfolio_id
        df['asset_id'] = asset_id

        # Reset the index to make 'date' a column
        df = df.reset_index()
        
        # TODO: include portfolio_id and asset_id in the output
        return df[['date', 'portfolio_id', 'asset_id', 'cash_flow_daily', 'cash_flow_cumulative']]

    def get_cash_flow_general(self, portfolio_ids: List[int], asset_ids: List[int], currency_id: int, start_date: date = date(1970, 1, 1), end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Calculate the value invested for multiple portfolios and assets over a specified date range.
        
        Args:
            session (Session): The database session.
            portfolio_ids (List[int]): The IDs of the portfolios to calculate value invested for.
            asset_ids (List[int]): The IDs of the assets to calculate value invested for.
            currency_id (int): The ID of the currency to convert the value to.
            start_date (date): The start date of the date range to consider.
            end_date (date): The end date of the date range to consider.

        Returns:
            pd.DataFrame: A melted DataFrame containing the cumulative value invested over time for all portfolio-asset combinations.
        """
        all_data = []

        for portfolio_id in portfolio_ids:
            for asset_id in asset_ids:
                cash_flow_df = self._get_cash_flow(portfolio_id, asset_id, currency_id, start_date, end_date)
                cash_flow_df['portfolio_id'] = portfolio_id
                cash_flow_df['asset_id'] = asset_id
                all_data.append(cash_flow_df)

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values(['date', 'portfolio_id', 'asset_id']).reset_index(drop=True)
            
            return df[['date', 'portfolio_id', 'asset_id', 'cash_flow_daily', 'cash_flow_cumulative']]
        else:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'cash_flow_daily', 'cash_flow_cumulative'])

    @with_session
    def get_cost_basis(self, session: Session, portfolio_id: int, asset_id: int, currency_id: int, start_date: date = date(1970, 1, 1), end_date: date = datetime.now().date()) -> pd.DataFrame:
        # TODO: recognize the limitations of this implementation. No guarantee that mulitple same-day transactions will be handled in correct order
        # test the behavior  
        """
        Calculate the cost basis of a specific asset for a given portfolio over a specified date range.
        
        Args:
            session (Session): The database session.
            portfolio_id (int): The ID of the portfolio.
            asset_id (int): The ID of the asset.
            currency_id (int): The ID of the currency to calculate the cost basis in.
            start_date (date): The start date of the date range to consider.
            end_date (date): The end date of the date range to consider.

        Returns:
            pd.DataFrame: A DataFrame containing the cost basis over time.
        """
        actions = self.db_access.get_buy_sell_actions_by_portfolio_id_asset_id(session, portfolio_id, asset_id)
        
        actions_df = pd.read_sql(actions.statement, session.bind)
        actions_df = actions_df.sort_values(by='date').reset_index(drop=True)

        actions_df['quantity'] = actions_df.apply(lambda row: row['quantity'] if row['action_type_name'] == 'buy' else -row['quantity'], axis=1)
        actions_df['cost'] = 0.0
        actions_df['cumulative_quantity'] = 0.0
        actions_df['cumulative_cost'] = 0.0

        for i, row in actions_df.iterrows():
            if row['action_type_name'] == 'buy':
                if currency_id != row['currency_id']:
                    conversion_rate = self.db_access.get_currency_conversion_on_date(session, row['currency_id'], currency_id, row['date'])
                    actions_df.at[i, 'cost'] = conversion_rate * row['price'] * row['quantity']
                else:
                    actions_df.at[i, 'cost'] = row['price'] * row['quantity']
            else:  # sell action
                # TODO: beware of multiple sales/buys in a day
                if actions_df.at[i-1, 'cumulative_quantity'] > 0:
                    actions_df.at[i, 'cost'] = row['quantity'] * actions_df.at[i-1, 'cost_basis']  # sale cost = quantity * cost basis

            actions_df.at[i, 'cumulative_quantity'] = actions_df['quantity'][:i+1].sum()
            actions_df.at[i, 'cumulative_cost'] = actions_df['cost'][:i+1].sum()
            
            if actions_df.at[i, 'cumulative_quantity'] > 0:
                actions_df.at[i, 'cost_basis'] = actions_df.at[i, 'cumulative_cost'] / actions_df.at[i, 'cumulative_quantity']
            else:
                actions_df.at[i, 'cost_basis'] = 0  # Reset cost basis when all assets are sold

        # Aggregate daily cost basis to a single value per day
        actions_df = actions_df.groupby(['date', 'portfolio_id', 'asset_id'])['cost_basis'].mean().reset_index()

        # Create a daily date range
        date_range = pd.date_range(start=max(start_date, actions_df['date'].min().date()), end=min(end_date, datetime.now().date()), freq='D')
        
        # Create a daily DataFrame
        df = pd.DataFrame(index=date_range)
        df.index.name = 'date'
        
        # Merge the action data with the daily DataFrame
        df = df.merge(actions_df[['date', 'portfolio_id', 'asset_id', 'cost_basis']], left_index=True, right_on='date', how='left')
        
        # Fill forward the values
        df['cost_basis'] = df['cost_basis'].ffill().astype(float)
        df['portfolio_id'] = portfolio_id
        df['asset_id'] = asset_id
        
        # Calculate the cost basis
        # df['cost_basis'] = df['cumulative_cost'] / df['cumulative_quantity']
        
        # Reset the index to make 'date' a column
        df = df.reset_index()
        
        return df[['date', 'portfolio_id', 'asset_id', 'cost_basis']]

    @with_session
    def get_unrealized_gain_loss(self, session: Session, portfolio_id: int, asset_id: int, currency_id: int, start_date: date = date(1970, 1, 1), end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Calculate the unrealized gain/loss for a specific asset in a given portfolio over a specified date range.

        Args:
            session (Session): The database session.
            portfolio_id (int): The ID of the portfolio.
            asset_id (int): The ID of the asset.
            currency_id (int): The ID of the currency to calculate the gain/loss in.
            start_date (date): The start date of the date range to consider.
            end_date (date): The end date of the date range to consider.

        Returns:
            pd.DataFrame: A DataFrame containing the unrealized gain/loss over time.
        """
        # Get cost basis
        cost_basis_df = self.get_cost_basis(session, portfolio_id, asset_id, currency_id, start_date, end_date)

        # Get holdings in base currency
        holdings_df = self._get_holdings_in_base_currency(session, portfolio_id, asset_id, currency_id, start_date, end_date)

        # Merge cost basis and holdings dataframes
        df = pd.merge(cost_basis_df, holdings_df, on='date', how='outer')
        df = df.sort_values('date').ffill()

        # Calculate unrealized gain/loss
        df['unrealized_gain_loss'] = df['value_in_currency'] - df['cumulative_cost']

        # Calculate unrealized gain/loss percentage
        df['unrealized_gain_loss_percentage'] = (df['unrealized_gain_loss'] / df['cumulative_cost']) * 100

        return df[['date', 'cumulative_quantity', 'cumulative_cost', 'cost_basis', 'value_in_currency', 'unrealized_gain_loss', 'unrealized_gain_loss_percentage']]

    def get_unrealized_gain_loss_general(self, portfolio_ids: List[int], asset_ids: List[int], currency_id: int, start_date: date = date(1970, 1, 1), end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Calculate the unrealized gain/loss for multiple portfolios and assets over a specified date range.

        Args:
            session (Session): The database session.
            portfolio_ids (List[int]): The IDs of the portfolios to calculate unrealized gain/loss for.
            asset_ids (List[int]): The IDs of the assets to calculate unrealized gain/loss for.
            currency_id (int): The ID of the currency to calculate the gain/loss in.
            start_date (date): The start date of the date range to consider.
            end_date (date): The end date of the date range to consider.

        Returns:
            pd.DataFrame: A melted DataFrame containing the unrealized gain/loss over time for all portfolio-asset combinations.
        """
        all_data = []

        for portfolio_id in portfolio_ids:
            for asset_id in asset_ids:
                unrealized_gain_loss_df = self.get_unrealized_gain_loss(portfolio_id, asset_id, currency_id, start_date, end_date)
                unrealized_gain_loss_df['portfolio_id'] = portfolio_id
                unrealized_gain_loss_df['asset_id'] = asset_id
                all_data.append(unrealized_gain_loss_df)

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values(['date', 'portfolio_id', 'asset_id']).reset_index(drop=True)
            
            return df[['date', 'portfolio_id', 'asset_id', 'cumulative_quantity', 'cumulative_cost', 'cost_basis', 'value_in_currency', 'unrealized_gain_loss', 'unrealized_gain_loss_percentage']]
        else:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'cumulative_quantity', 'cumulative_cost', 'cost_basis', 'value_in_currency', 'unrealized_gain_loss', 'unrealized_gain_loss_percentage'])

    def get_time_weighted_return_general(self, portfolio_ids: List[int], asset_ids: List[int], currency_id: int, start_date: date = date(1970, 1, 1), end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Calculate the time-weighted return for multiple portfolios and assets over a specified date range.

        Args:
            session (Session): The database session.
            portfolio_ids (List[int]): The IDs of the portfolios to calculate time-weighted return for.
            asset_ids (List[int]): The IDs of the assets to calculate time-weighted return for.
            currency_id (int): The ID of the currency to calculate the return in.
            start_date (date): The start date of the date range to consider.
            end_date (date): The end date of the date range to consider.

        Returns:
            pd.DataFrame: A DataFrame containing the time-weighted return over time.
        """

        # Get the unrealized gain/loss data for all portfolio-asset combinations
        holdings_value = self.get_holdings_in_base_currency_general(portfolio_ids=portfolio_ids, asset_ids=asset_ids, currency_id=currency_id, start_date=start_date, end_date=end_date)
        cash_flow = self.get_cash_flow_general(portfolio_ids=portfolio_ids, asset_ids=asset_ids, currency_id=currency_id, start_date=start_date, end_date=end_date)

        # merge
        data = pd.merge(left=cash_flow, right=holdings_value, on=['date', 'portfolio_id', 'asset_id'], how='outer')

        # aggregate on date
        data_agg = self.aggregate_on_date(df=data, columns=['value_invested_discrete', 'value_holdings'])
        data_agg.rename(columns={'value_invested_discrete': 'cash_flows'}, inplace=True)

        # Calculate periodic (daily) returns
        adjusted_daily_returns = (data_agg['value_holdings'][1:].values - data_agg['cash_flows'][1:].values) / data_agg['value_holdings'][:-1].values
        adjusted_daily_returns = np.insert(adjusted_daily_returns, 0, 1)

        # Calculate cumulative returns time series, adn add to data_agg
        cumulative_returns = np.cumprod(adjusted_daily_returns)
        data_agg['daily_adjusted_return'] = adjusted_daily_returns
        data_agg['time_weighted_return'] = cumulative_returns

        return data_agg[['date', 'daily_adjusted_return', 'time_weighted_return']]

    def get_sharpe_ratio(self, portfolio_ids: List[int], asset_ids: List[int], currency_id: int, risk_free_rate: float = 0.0, rolling_window_periods: int = 180, periods_per_year: int = 365, start_date: date = date(1970, 1, 1), end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Calculate the Sharpe ratio based on the time-weighted return over a rolling window.

        Args:
            twr_df (pd.DataFrame): DataFrame containing the time-weighted return data.
            risk_free_rate (float): The risk-free rate (default: 0.0).
            rolling_window (int): The number of days for the rolling window calculation (default: 30).
            periods_per_year (int): Number of periods per year (default: 252 for daily data).

        Returns:
            pd.DataFrame: A DataFrame containing the rolling Sharpe ratio.
        """
        twr_df = self.get_time_weighted_return_general(portfolio_ids=portfolio_ids, asset_ids=asset_ids, currency_id=currency_id, start_date=start_date, end_date=end_date)

        # Calculate excess return
        excess_return = twr_df['time_weighted_return']

        # Calculate rolling mean and standard deviation
        rolling_annualized_return = excess_return.rolling(window=rolling_window_periods).apply(lambda x: x.iloc[-1] - x.iloc[0])*periods_per_year/rolling_window_periods
        rolling_excess_annualized_return = rolling_annualized_return-risk_free_rate
        rolling_std = excess_return.rolling(window=rolling_window_periods).std()

        # Calculate annualized Sharpe ratio
        rolling_sharpe_ratio = rolling_excess_annualized_return / rolling_std

        # Create a DataFrame with the rolling Sharpe ratio
        sharpe_df = pd.DataFrame({
            'date': twr_df['date'],
            'rolling_sharpe_ratio': rolling_sharpe_ratio
        })

        return sharpe_df[['date', 'rolling_sharpe_ratio']]

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
    # TODO: DONE ~~time-weighted return~~
    # TODO: time range on calculations? 
    # TODO: realized gain/loss, each sale 1 line - fields: realized gain/loss, implied tax
    # TODO: benchmark comparison - time-weighted return
    # TODO: DONE ~~sharpe ratio - on what? (time-weighted return - like a piecewise return - break down return into periods of constant value invested)~~

    metric_service = MetricService(DatabaseAccess())


    # test the date range
    out = metric_service._get_holdings_in_base_currency(portfolio_id=1, asset_id=4, currency_id=1, start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))
    out = metric_service._get_holdings_in_base_currency(portfolio_id=1, asset_id=4, currency_id=1, start_date=date(2023, 1, 1), end_date=date(2024, 1, 1))
    out = metric_service._get_holdings_in_base_currency(portfolio_id=1, asset_id=4, currency_id=1)

    out = metric_service.get_holdings_in_base_currency_general(portfolio_ids=[1,2], asset_ids=[4,5,6], currency_id=1, start_date=date(2023, 1, 1), end_date=date(2023, 1, 31)).groupby('date')[['value_holdings']].sum().reset_index()
    out = metric_service.get_holdings_in_base_currency_general(portfolio_ids=[1,2], asset_ids=[4,5,6], currency_id=1, start_date=date(2023, 1, 1), end_date=date(2024, 1, 1)).groupby('date')[['value_holdings']].sum().reset_index()
    out = metric_service.get_holdings_in_base_currency_general(portfolio_ids=[1,2], asset_ids=[4,5,6], currency_id=1).groupby('date')[['value_holdings']].sum().reset_index()

    out = metric_service._get_cash_flow(portfolio_id=1, asset_id=4, currency_id=1, start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))
    out = metric_service._get_cash_flow(portfolio_id=1, asset_id=4, currency_id=1, start_date=date(2023, 1, 1), end_date=date(2024, 1, 1))
    out = metric_service._get_cash_flow(portfolio_id=1, asset_id=4, currency_id=1)

    out = metric_service.get_cash_flow_general(portfolio_ids=[1,2], asset_ids=[4,5,6], currency_id=1, start_date=date(2023, 1, 1), end_date=date(2023, 1, 31)).groupby('date')[['cash_flow_daily', 'cash_flow_cumulative']].sum().reset_index()
    out = metric_service.get_cash_flow_general(portfolio_ids=[1,2], asset_ids=[4,5,6], currency_id=1, start_date=date(2023, 1, 1), end_date=date(2024, 1, 1)).groupby('date')[['cash_flow_daily', 'cash_flow_cumulative']].sum().reset_index()
    out = metric_service.get_cash_flow_general(portfolio_ids=[1,2], asset_ids=[4,5,6], currency_id=1).groupby('date')[['cash_flow_daily', 'cash_flow_cumulative']].sum().reset_index()

    out = metric_service.get_cost_basis(portfolio_id=1, asset_id=4, currency_id=1, start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))
    out = metric_service.get_cost_basis(portfolio_id=1, asset_id=4, currency_id=1, start_date=date(2023, 1, 1), end_date=date(2024, 1, 31))
    out = metric_service.get_cost_basis(portfolio_id=1, asset_id=4, currency_id=1)

    # continue testing

    start_time = time.time()
    result_sr = metric_service.get_sharpe_ratio(portfolio_ids=[1,2], asset_ids=[4,5,6], currency_id=1, risk_free_rate=0.0, rolling_window_periods=90, periods_per_year=365)
    end_time = time.time()
    print(f"Runtime for get_sharpe_ratio: {end_time - start_time:.3f} seconds")

    start_time = time.time()
    result_twr = metric_service.get_time_weighted_return_general(portfolio_ids=[1,2], asset_ids=[4,5,6], currency_id=1)
    end_time = time.time()
    print(f"Runtime for get_time_weighted_return_general: {end_time - start_time:.3f} seconds")

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
