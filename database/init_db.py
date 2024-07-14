
from datetime import datetime, timedelta

from database.access import DatabaseAccess
from database.entities import ActionType, Asset
from services.yfinance_service import YFinanceService
from config import action_types, assets


def initialize_database():
    db_access = DatabaseAccess()
    db_access.init_db()
    db_access.insert_if_not_exists(ActionType, action_types)
    db_access.insert_if_not_exists(Asset, assets)



if __name__ == "__main__":
    initialize_database()
    print('done')