
from database.access_v2 import DatabaseAccess
from database.entities import ActionType, Asset, Portfolio
from config import action_types, assets, portfolios

db = DatabaseAccess()

@db.with_session
def initialize_database(session):
    db.init_db()
    db.insert_if_not_exists(session, ActionType, action_types)
    db.insert_if_not_exists(session, Asset, assets)
    db.insert_if_not_exists(session, Portfolio, portfolios)


if __name__ == "__main__":
    initialize_database()
    print('done')