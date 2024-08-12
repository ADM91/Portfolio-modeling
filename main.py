
from database.access import DatabaseAccess
from database.init_db import initialize_database
from services.yfinance_service import YFinanceService
from services.action_service import ActionService 
from services.portfolio_service import PortfolioService


def startup():

    db_access = DatabaseAccess()
    initialize_database()
    
    # YFinanceService get asset time series data
    yfinance_service = YFinanceService(db_access)
    yfinance_service.update_db_with_asset_data()

    # ActionService get action data
    action_service = ActionService(db_access)
    actions = action_service.read_actions_from_excel_to_action_list("_junk/test_action_data.xlsx")
    action_service.insert_actions(actions)  # TODO: this reinserts existing actions

    # PortfolioService update portfolio holdings data
    portfolio_service = PortfolioService(db_access)
    portfolio_service.process_actions()
    
    # TODO next step: mechanism to produce time series for portfolio holdings
        # Need selection of base currency and conversion of all asset values to the base currency.
        # create in kind time series


if __name__ == "__main__":
    startup()