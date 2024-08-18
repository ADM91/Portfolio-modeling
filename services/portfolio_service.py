
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

from database.entities import PortfolioHoldingsTimeSeries, Asset, Portfolio


class PortfolioService:
    """
    Service class for managing portfolios and their activities - updates in-kind holdings in database
    """
    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access

    def merge_time_series(self, ts_1, ts_2):
        ts_1 = pd.merge(left=ts_1, right=ts_2, on='date', how='outer')
        ts_1['quantity'] = ts_1[['quantity_x', 'quantity_y']].sum(axis=1)
        ts_1.drop(columns=['quantity_x', 'quantity_y'], inplace=True)
        return ts_1

    def process_actions(self):

        # Get unprocessed actions
        unprocessed_actions = self.db_access.get_unprocessed_actions()

        # Process unprocessed actions, update portfolio holdings  
        for action in unprocessed_actions:

            # run transaction, update portfolio holdings and action as processed
            self.db_access.update_portfolio_holdings_and_action(action)
        
        # Update time series
        self.update_holdings_time_series()

        return
    
    def generate_holdings_time_series(self, start_date: datetime, end_date: datetime):
        with self.db_access.session_scope() as session:
            portfolios = session.query(Portfolio).all()
            assets = session.query(Asset).all()
            
            for portfolio in portfolios:
                holdings_series = self._generate_portfolio_holdings(session, portfolio.id, assets, start_date, end_date)
                self._store_holdings_time_series(session, portfolio.id, holdings_series)

    def _generate_portfolio_holdings(self, session, portfolio_id: int, assets: List[Asset], start_date: datetime, end_date: datetime) -> Dict[int, pd.DataFrame]:
        holdings_series = {}
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for asset in assets:
            actions = self.db_access.get_portfolio_asset_actions(session, portfolio_id, asset.id)
            if not actions:
                continue

            df = pd.DataFrame(index=date_range, columns=['quantity'])
            df['quantity'] = 0

            for action in actions:
                if action.action_type.name in ('buy', 'dividend'):
                    df.loc[action.date:, 'quantity'] += action.quantity
                elif action.action_type.name == 'sell':
                    df.loc[action.date:, 'quantity'] -= action.quantity

            holdings_series[asset.id] = df.fillna(method='ffill').fillna(0)

        return holdings_series

    def _store_holdings_time_series(self, session, portfolio_id: int, holdings_series: Dict[int, pd.DataFrame]):
        for asset_id, df in holdings_series.items():
            for date, row in df.iterrows():
                holding = PortfolioHoldingsTimeSeries(
                    portfolio_id=portfolio_id,
                    asset_id=asset_id,
                    date=date,
                    quantity=row['quantity']
                )
                session.merge(holding)
    # def generate_holdings_time_series(self, start_date: datetime, end_date: datetime):
    #     portfolios = self.db_access.get_all_portfolios()
    #     assets = self.db_access.get_all_assets()
        
    #     for portfolio in portfolios:
    #         holdings_series = self._generate_holdings_time_series(portfolio.id, assets, start_date, end_date)
    #         self._store_holdings_time_series(portfolio.id, holdings_series)

    # def _generate_holdings_time_series(self, portfolio_id: int, assets: List[Asset], start_date: datetime, end_date: datetime) -> Dict[int, pd.DataFrame]:
    #     holdings_series = {}
    #     date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
    #     for asset in assets:
    #         actions = self.db_access.get_portfolio_asset_actions(session, portfolio_id, asset.id)
    #         if not actions:
    #             continue

    #         df = pd.DataFrame(index=date_range, columns=['quantity'])
    #         df['quantity'] = 0

    #         for action in actions:
    #             if action.action_type.name in ('buy', 'dividend'):
    #                 df.loc[action.date:, 'quantity'] += action.quantity
    #             elif action.action_type.name == 'sell':
    #                 df.loc[action.date:, 'quantity'] -= action.quantity

    #         holdings_series[asset.id] = df.fillna(method='ffill').fillna(0)

    #     return holdings_series

    # def _store_holdings_time_series(self, portfolio_id: int, holdings_series: Dict[int, pd.DataFrame]):
    #     with self.db_access.session_scope() as session:
    #         for asset_id, df in holdings_series.items():
    #             for date, row in df.iterrows():
    #                 holding = PortfolioHoldingsTimeSeries(
    #                     portfolio_id=portfolio_id,
    #                     asset_id=asset_id,
    #                     date=date,
    #                     quantity=row['quantity']
    #                 )
    #                 session.merge(holding)

    def update_holdings_time_series(self):
        last_update = self.db_access.get_last_holdings_time_series_update()
        end_date = datetime.now().date()
        
        if last_update:
            start_date = last_update + timedelta(days=1)
        else:
            start_date = self.db_access.get_earliest_action_date()

        if start_date < end_date:
            self.generate_holdings_time_series(start_date, end_date)

    def regenerate_holdings_time_series(self):
        self.db_access.clear_holdings_time_series()
        earliest_date = self.db_access.get_earliest_action_date()
        end_date = datetime.now().date()
        self.generate_holdings_time_series(earliest_date, end_date)

    def insert_portfolio(self, portfolio):
        # Insert portfolio into database
        pass
    
    def update_portfolio(self, portfolio_id, name, owner):
        # Update portfolio in database
        pass

    def delete_portfolio(self, portfolio_id):
        # Delete portfolio from database
        pass
    
    def update_portfolio_holdings(self, portfolio_id, asset_id, quantity, date):
        # Update portfolio holdings in database
        pass

    def get_portfolio_holdings(self, portfolio_id, start_date, end_date):
        # Fetch portfolio holdings from database
        pass

    def delete_portfolio_holdings(self, portfolio_id):
        # Delete portfolio holdings from database
        pass



if __name__ == "__main__":

    db_access = DatabaseAccess()
    portfolio_service = PortfolioService(db_access)
    portfolio_service.process_actions()

    print('done')



    # def get_portfolio_metrics(self, portfolio_id, start_date=None, end_date=None):
    #     with self.db_manager.get_connection() as conn:
    #         query = '''
    #             SELECT date, total_value, daily_return, cumulative_return, volatility, sharpe_ratio
    #             FROM portfolio_metrics
    #             WHERE portfolio_id = ?
    #         '''
    #         params = [portfolio_id]
    #         if start_date:
    #             query += ' AND date >= ?'
    #             params.append(start_date)
    #         if end_date:
    #             query += ' AND date <= ?'
    #             params.append(end_date)
    #         query += ' ORDER BY date'
            
    #         df = pd.read_sql_query(query, conn, params=params)
    #         df['date'] = pd.to_datetime(df['date'])
    #         df.set_index('date', inplace=True)
    #         return df
        

    # def process_activity(self, activity):
    #     """
    #     Processes a single action by updating or creating portfolios and assets accordingly.
    #     """
    #     # portfolio_name = activity['PortfolioName']
    #     # asset_name = activity['AssetName']
    #     # action_type = activity['ActionType']
    #     # date = activity['Date']
    #     # amount = activity['Amount']
    #     # currency = activity['Currency']
    #     # remarks = activity.get('Remarks', '')

    #     # Retrieve or create the Portfolio object
    #     portfolio = self.portfolios.get(activity.Account)
    #     if portfolio is None:
    #         portfolio = Portfolio(activity.Account)
    #         self.portfolios[activity.Account] = portfolio

    #     # Retrieve or create the Asset object
    #     asset = portfolio.get_asset(activity.Asset)
    #     if asset is None:
    #         asset = Asset(activity.Asset, ticker_symbol='')  # Assuming ticker_symbol or similar identifier is available
    #         portfolio.add_asset(asset)

    #     # Depending on action type, update the asset (and portfolio if necessary)
    #     # if action_type == 'Transaction':
    #     #     asset.add_transaction(Act(date=date, amount=amount, currency=currency, remarks=remarks))
    #     # Add more conditions for different action types like 'Dividend', 'PriceChange', etc.

    #     # Optionally, update portfolio-level metrics or balances if necessary

    # def update_portfolio_analytics(self):
    #     for portfolio_id, portfolio in self.portfolios.items():
    #         activities_df = self._get_portfolio_activities(portfolio_id)
    #         self._update_portfolio_holdings(portfolio_id, activities_df)
    #         self._calculate_portfolio_metrics(portfolio_id)

    # def _update_portfolio_holdings(self, portfolio_id, activities_df):
    #     holdings = self._calculate_daily_holdings(activities_df)
        
        # with self.db_manager.get_connection() as conn:
        #     holdings.to_sql('portfolio_holdings', conn, if_exists='replace', index=False)

    # def _calculate_daily_holdings(self, activities_df):
    #     # Logic to transform activities into daily holdings
    #     pass

    # def _calculate_portfolio_metrics(self, portfolio_id, start_date=None, end_date=None):
    #     holdings = self._get_portfolio_holdings(portfolio_id, start_date, end_date)
    #     prices = self._get_asset_prices(portfolio_id, start_date, end_date)

    #     portfolio_value = self._calculate_portfolio_value(holdings, prices)

    #     metrics = pd.DataFrame(index=portfolio_value.index)
    #     metrics['total_value'] = portfolio_value
    #     metrics['daily_return'] = portfolio_value.pct_change()
    #     metrics['cumulative_return'] = (1 + metrics['daily_return']).cumprod() - 1
    #     metrics['volatility'] = metrics['daily_return'].rolling(window=30).std() * np.sqrt(252)
    #     risk_free_rate = 0.02
    #     metrics['sharpe_ratio'] = (metrics['daily_return'].mean() * 252 - risk_free_rate) / (metrics['daily_return'].std() * np.sqrt(252))

    #     with self.db_manager.get_connection() as conn:
    #         metrics.to_sql('portfolio_metrics', conn, if_exists='replace', index=False)

    # def process_actions(self):

    #     # Get actions
    #     actions_unprocessed = self.db_access.get_unprocessed_actions()

    #     # Holdings time series are generated using pandas dataframes
    #     holdings_dfs = {}

    #     # Process unprocessed actions, update portfolio holdings  
    #     # TODO need to track action processing so i can update is_processed flag  
    #     for action in actions_unprocessed:
            
    #         # create df if it doesnt exist
    #         if holdings_dfs.get(action.portfolio_id) is None:
    #             holdings_dfs[action.portfolio_id] = {}
    #         if holdings_dfs[action.portfolio_id].get(action.asset_id) is None:
    #             holdings_dfs[action.portfolio_id][action.asset_id] = None

    #         # if action is buy or dividend
    #         if action.action_type_id in (1,3):  

    #             # create series
    #             in_kind_ts = pd.DataFrame(pd.date_range(start=action.date, end=datetime.now(), freq='D'), columns=['date'])
    #             in_kind_ts['quantity'] = action.quantity  

    #             # if there is alread a series under this portfolio and asset, aggregate them
    #             if holdings_dfs[action.portfolio_id][action.asset_id] is not None:
    #                 # merge dataframes
    #                 holdings_dfs[action.portfolio_id][action.asset_id] = self.merge_time_series(holdings_dfs[action.portfolio_id][action.asset_id], in_kind_ts)

    #             else:
    #                 holdings_dfs[action.portfolio_id][action.asset_id] = in_kind_ts

    #         elif action.action_type_id == 2:

    #             # create series (sell will never start from 0)
    #             in_kind_ts = pd.DataFrame(pd.date_range(start=action.date, end=datetime.now(), freq='D'), columns=['date'])
    #             in_kind_ts['quantity'] = -action.quantity

    #             # merge dataframes
    #             holdings_dfs[action.portfolio_id][action.asset_id] = self.merge_time_series(holdings_dfs[action.portfolio_id][action.asset_id], in_kind_ts)
  

    #     # Update portfolio holdings in database
    #     for portfolio_id in holdings_dfs:
    #         for asset_id in holdings_dfs[portfolio_id]:
    #             self.db_access.update_portfolio_holdings(portfolio_id, asset_id, holdings_dfs[portfolio_id][asset_id], datetime.now())

    #     return
