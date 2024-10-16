
import os

from database.access import DatabaseAccess
from database.init_db import initialize_database
from services.yfinance_service import YFinanceService
from services.action_service import ActionService
from services.metric_service import MetricService
from services.portfolio_service import PortfolioService



def startup():

    # Initialize the database
    db_access = DatabaseAccess()
    initialize_database(db_access)
    
    # YFinanceService get asset time series data
    yfinance_service = YFinanceService(db_access)
    yfinance_service.update_db_with_asset_data()

    # ActionService get action data
    action_service = ActionService(db_access)
    actions = action_service.read_actions_from_excel(os.environ.get("PATH_ACTIONS"))
    action_service.insert_actions(actions)  # TODO: this reinserts existing actions
    action_service.process_actions()
    action_service.update_holdings_time_series_to_current_day()

    # MetricService calculate metrics
    metric_service = MetricService(db_access)

    # # PortfolioService update portfolio holdings data
    # portfolio_service = PortfolioService(db_access)
    # portfolio_service.process_actions()


if __name__ == "__main__":
    startup()