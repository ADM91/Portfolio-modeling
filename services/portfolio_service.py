
import pandas as pd
from pydantic import ValidationError

from models.asset import Asset  # Assuming you have an Asset class defined as previously discussed
from models.portfolio import Portfolio  # Assuming you have a Portfolio class that manages a collection of Assets
from models.activity import Activity
from database.access import DatabaseAccess


class PortfolioService:
    """
    Service class for managing portfolios and their activities - updates in-kind holdings in database
    """
    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access


    def process_actions(self, actions : list[Activity]):

        # For each activiy

        return

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


