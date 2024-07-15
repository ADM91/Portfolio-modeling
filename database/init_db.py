
from database.access import DatabaseAccess
from database.entities import ActionType, Asset, Portfolio
from config import action_types, assets, portfolios


def initialize_database():
    db_access = DatabaseAccess()
    db_access.init_db()
    db_access.insert_if_not_exists(ActionType, action_types)
    db_access.insert_if_not_exists(Asset, assets)
    db_access.insert_if_not_exists(Portfolio, portfolios)


if __name__ == "__main__":
    initialize_database()
    print('done')