
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
from sqlalchemy.orm import Session

from database.entities import PortfolioHoldingsTimeSeries, Asset, Portfolio, Action
from database.access import with_session, session_scope, DatabaseAccess


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

    
    # def update_holdings_time_series(self, session: Session):
    #     last_update = self.db_access.get_last_holdings_time_series_update(session)
    #     end_date = datetime.now().date()
        
    #     if last_update:
    #         start_date = last_update + timedelta(days=1)
    #     else:
    #         start_date = self.db_access.get_earliest_action_date(session)

    #     if start_date < end_date:
    #         self.generate_holdings_time_series(session, start_date, end_date)

    # def generate_holdings_time_series(self, session: Session, start_date: datetime, end_date: datetime):
        # portfolios = self.db_access.get_portfolios(session)
        
        # for portfolio in portfolios:
        #     assets = self.db_access.get_portfolio_assets(session, portfolio.id)
        #     holdings_series = self._generate_portfolio_holdings(session, portfolio.id, assets, start_date, end_date)
        #     self._store_holdings_time_series(session, portfolio.id, holdings_series)

    # def _generate_holdings_time_series(self, action: Action) -> Dict[int, pd.DataFrame]:

    #     date_range = pd.date_range(start=action.date, end=datetime.now().date(), freq='D')
    #     df = pd.DataFrame(index=date_range, columns=['quantity'])
    #     df['quantity'] = 0

    #     if action.action_type.name in ('buy', 'dividend'):
    #         df.loc[action.date:, 'quantity'] += action.quantity
    #     elif action.action_type.name == 'sell':
    #         df.loc[action.date:, 'quantity'] -= action.quantity

    #     return df


    # def _generate_portfolio_holdings_old(self, session: Session, portfolio_id: int, assets: List[Asset], start_date: datetime, end_date: datetime) -> Dict[int, pd.DataFrame]:
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

    # def _store_holdings_time_series(self, session: Session, portfolio_id: int, holdings_series: Dict[int, pd.DataFrame]):
    #     holdings_data = []
    #     for asset_id, df in holdings_series.items():
    #         for date, row in df.iterrows():
    #             holdings_data.append({
    #                 'portfolio_id': portfolio_id,
    #                 'asset_id': asset_id,
    #                 'date': date,
    #                 'quantity': row['quantity']
    #             })
        
    #     self.db_access.store_holdings_time_series(session, holdings_data)

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
