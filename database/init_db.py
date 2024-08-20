
from database.access import with_session, DatabaseAccess
from database.entities import ActionType, Asset, Portfolio
from config import action_types, assets, portfolios


@with_session
def initialize_database(session, db: DatabaseAccess):
    db.init_db()
    db.insert_if_not_exists(session, ActionType, action_types)
    db.insert_if_not_exists(session, Asset, assets)
    db.insert_if_not_exists(session, Portfolio, portfolios)


if __name__ == "__main__":
    db = DatabaseAccess()
    initialize_database(db)
    print('done')