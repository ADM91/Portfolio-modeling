import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, joinedload
from typing import List, Callable

from database.entities import Action, Portfolio, Asset, PortfolioHoldingsTimeSeries

# Create engine and session factory
engine = create_engine(os.environ.get('DATABASE_URL'))
SessionFactory = sessionmaker(bind=engine)

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def with_session(func: Callable):
    """Decorator to automatically handle session creation and cleanup."""
    def wrapper(*args, **kwargs):
        with session_scope() as session:
            return func(session, *args, **kwargs)
    return wrapper

def get_portfolio_asset_actions(session, portfolio_id: int, asset_id: int) -> List[Action]:
    return session.query(Action).options(
        joinedload(Action.action_type)
    ).filter(
        Action.portfolio_id == portfolio_id,
        Action.asset_id == asset_id
    ).order_by(Action.date).all()

def get_all_portfolios(session) -> List[Portfolio]:
    return session.query(Portfolio).all()

def get_all_assets(session) -> List[Asset]:
    return session.query(Asset).all()

def merge_portfolio_holdings_time_series(session, holding: PortfolioHoldingsTimeSeries):
    session.merge(holding)
