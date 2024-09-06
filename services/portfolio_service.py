
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
from sqlalchemy.orm import Session

from database.entities import PortfolioHoldingsTimeSeries, Asset, Portfolio, Action
from database.access import with_session, session_scope, DatabaseAccess


class PortfolioService:
    """
    Service class for managing portfolios and their activities
    """
    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access

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

    print('done')
