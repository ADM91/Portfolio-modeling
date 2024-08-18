
import os
from contextlib import contextmanager
from sqlalchemy import create_engine, select, and_, func
from sqlalchemy.orm import sessionmaker, joinedload
from typing import List, Dict, Optional, Callable
from datetime import datetime
from functools import wraps

from database.entities import *


class DatabaseAccess:
    def __init__(self):
        self.engine = create_engine(os.environ.get('DATABASE_URL'))
        self.SessionFactory = sessionmaker(bind=self.engine)

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def with_session(self, func: Callable):
        """Decorator to automatically handle session creation and cleanup."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.session_scope() as session:
                if args and hasattr(args[0], func.__name__):
                    # This is likely a method call, insert session after self
                    return func(args[0], session, *args[1:], **kwargs)
                else:
                    # This is likely a regular function call
                    return func(session, *args, **kwargs)
        return wrapper

    def init_db(self):
        Base.metadata.create_all(self.engine)

    def insert_if_not_exists(self, session, model, data: List[Dict], filter_fields: Optional[List[str]] = None):
        for item in data:
            if filter_fields:
                filter_dict = {field: item[field] for field in filter_fields if field in item}
                exists = session.query(model).filter_by(**filter_dict).first()
            else: 
                exists = session.query(model).filter_by(**item).first()
            if not exists:
                obj = model(**item)
                session.add(obj)

    def add_asset(self, session, ticker: str, name: str):
        existing_asset = session.execute(
            select(Asset).where(Asset.ticker == ticker)
        ).scalar_one_or_none()

        if existing_asset:
            if existing_asset.name != name:
                existing_asset.name = name
            return existing_asset
        else:
            new_asset = Asset(ticker=ticker, name=name)
            session.add(new_asset)
            session.flush()
            return new_asset

    def add_price_history(self, session: session_scope, ticker: str, price_history: List[Dict]):
        asset = session.execute(select(Asset).where(Asset.ticker == ticker)).scalar_one_or_none()
        if not asset:
            raise ValueError(f"Asset with ticker {ticker} not found")

        min_date = min(entry['date'] for entry in price_history)

        existing_records = session.execute(
            select(PriceHistory)
            .where(
                and_(
                    PriceHistory.asset_id == asset.id,
                    PriceHistory.date >= min_date
                )
            )
        ).all()

        existing_dates = {record[0].date.date(): record[0].id for record in existing_records}

        to_insert = []
        to_update = []

        for entry in price_history:
            date = entry['date'].date() if isinstance(entry['date'], datetime) else entry['date']
            if date in existing_dates:
                to_update.append({
                    'id': existing_dates[date],
                    'asset_id': asset.id,
                    'date': date,
                    'open': entry['open'],
                    'high': entry['high'],
                    'low': entry['low'],
                    'close': entry['close'],
                    'volume': entry['volume']
                })
            else:
                to_insert.append(PriceHistory(
                    asset_id=asset.id,
                    date=date,
                    open=entry['open'],
                    high=entry['high'],
                    low=entry['low'],
                    close=entry['close'],
                    volume=entry['volume']
                ))

        if to_insert:
            session.bulk_save_objects(to_insert)

        if to_update:
            session.bulk_update_mappings(PriceHistory, to_update)

    def get_all_assets(self, session) -> List[Asset]:
        return session.query(Asset).all()

    def get_all_currencies(self, session) -> List[Asset]:
        return session.query(Asset).filter(Asset.is_currency == True).all()

    def get_last_price_date(self, session, ticker: str) -> Optional[datetime]:
        asset = session.query(Asset).filter(Asset.ticker == ticker).first()
        if asset:
            last_price = session.query(PriceHistory).filter(PriceHistory.asset_id == asset.id).order_by(PriceHistory.date.desc()).first()
            return last_price.date if last_price else None
        return None

    def get_action_type_by_name(self, session, name: str) -> Optional[ActionType]:
        return session.query(ActionType).filter(ActionType.name == name).first()

    def get_asset_by_code(self, session, code: str) -> Optional[Asset]:
        return session.query(Asset).filter(Asset.code == code).first()

    def get_portfolio_by_name(self, session, name: str) -> Optional[Portfolio]:
        return session.query(Portfolio).filter(Portfolio.name == name).first()

    def get_unprocessed_actions(self, session) -> List[Action]:
        return session.query(Action).filter(Action.is_processed == False).order_by(Action.date).all()

    def update_portfolio_holdings_and_action(self, session: session_scope, action: Action):
        quantity_old = (
            session.query(PortfolioHolding.quantity_new)
            .join(Action)
            .filter(
                (Action.portfolio_id == action.portfolio_id) &
                (Action.asset_id == action.asset_id)
            )
            .order_by(PortfolioHolding.date.desc())
            .first()
        )

        quantity_old = quantity_old[0] if quantity_old else 0

        if action.action_type_id in (ActionTypeEnum.buy.value, ActionTypeEnum.dividend.value):
            quantity_change = action.quantity
        elif action.action_type_id == ActionTypeEnum.sell.value:
            quantity_change = -action.quantity
        else:
            quantity_change = 0

        session.add(PortfolioHolding(
            action_id=action.asset_id,
            portfolio_id=action.portfolio_id,
            asset_id=action.asset_id,
            quantity_change=quantity_change,
            quantity_new=quantity_old+quantity_change,
            date=action.date,
            action=action
        ))

        session.query(Action).filter(Action.id == action.id).update({'is_processed': True})

    def get_last_holdings_time_series_update(self, session) -> Optional[datetime]:
        last_update = session.query(func.max(PortfolioHoldingsTimeSeries.date)).scalar()
        return last_update.date() if last_update else None

    def clear_holdings_time_series(self, session):
        session.query(PortfolioHoldingsTimeSeries).delete()

    def get_earliest_action_date(self, session) -> datetime:
        earliest_date = session.query(func.min(Action.date)).scalar()
        return earliest_date.date() if earliest_date else datetime.now().date()

    def get_all_portfolios(self, session) -> List[Portfolio]:
        return session.query(Portfolio).all()

    def get_portfolio_asset_actions(self, session, portfolio_id: int, asset_id: int) -> List[Action]:
        return session.query(Action).options(
            joinedload(Action.action_type)
        ).filter(
            Action.portfolio_id == portfolio_id,
            Action.asset_id == asset_id
        ).order_by(Action.date).all()

    def merge_portfolio_holdings_time_series(self, session, holdings: List[PortfolioHoldingsTimeSeries]):
        for holding in holdings:
            existing = session.query(PortfolioHoldingsTimeSeries).filter(
                PortfolioHoldingsTimeSeries.portfolio_id == holding.portfolio_id,
                PortfolioHoldingsTimeSeries.asset_id == holding.asset_id,
                PortfolioHoldingsTimeSeries.date == holding.date
            ).first()
            if existing:
                existing.quantity = holding.quantity
            else:
                session.add(holding)

    def get_portfolio_holdings_time_series(self, session, portfolio_id: int, asset_id: int, start_date: datetime, end_date: datetime) -> List[PortfolioHoldingsTimeSeries]:
        return session.query(PortfolioHoldingsTimeSeries).filter(
            PortfolioHoldingsTimeSeries.portfolio_id == portfolio_id,
            PortfolioHoldingsTimeSeries.asset_id == asset_id,
            PortfolioHoldingsTimeSeries.date.between(start_date, end_date)
        ).order_by(PortfolioHoldingsTimeSeries.date).all()




# # Create engine and session factory
# engine = create_engine(os.environ.get('DATABASE_URL'))
# SessionFactory = sessionmaker(bind=engine)

# @contextmanager
# def session_scope():
#     """Provide a transactional scope around a series of operations."""
#     session = SessionFactory()
#     try:
#         yield session
#         session.commit()
#     except:
#         session.rollback()
#         raise
#     finally:
#         session.close()

# def with_session(func: Callable):
#     """Decorator to automatically handle session creation and cleanup."""
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         with session_scope() as session:
#             if args and hasattr(args[0], func.__name__):
#                 # This is likely a method call, insert session after self
#                 return func(args[0], session, *args[1:], **kwargs)
#             else:
#                 # This is likely a regular function call
#                 return func(session, *args, **kwargs)
#     return wrapper


# def init_db(session: session_scope):
#     Base.metadata.create_all(engine)


# def insert_if_not_exists(session: session_scope, model, data: List[Dict], filter_fields: Optional[List[str]] = None):
#     for item in data:
#         if filter_fields:
#             filter_dict = {field: item[field] for field in filter_fields if field in item}
#             exists = session.query(model).filter_by(**filter_dict).first()
#         else: 
#             exists = session.query(model).filter_by(**item).first()
#         if not exists:
#             obj = model(**item)
#             session.add(obj)


# def add_asset(session: session_scope, ticker: str, name: str):
#     existing_asset = session.execute(
#         select(Asset).where(Asset.ticker == ticker)
#     ).scalar_one_or_none()

#     if existing_asset:
#         if existing_asset.name != name:
#             existing_asset.name = name
#         return existing_asset
#     else:
#         new_asset = Asset(ticker=ticker, name=name)
#         session.add(new_asset)
#         session.flush()
#         return new_asset


# def add_price_history(session: session_scope, ticker: str, price_history: List[Dict]):
#     asset = session.execute(select(Asset).where(Asset.ticker == ticker)).scalar_one_or_none()
#     if not asset:
#         raise ValueError(f"Asset with ticker {ticker} not found")

#     min_date = min(entry['date'] for entry in price_history)

#     existing_records = session.execute(
#         select(PriceHistory)
#         .where(
#             and_(
#                 PriceHistory.asset_id == asset.id,
#                 PriceHistory.date >= min_date
#             )
#         )
#     ).all()

#     existing_dates = {record[0].date.date(): record[0].id for record in existing_records}

#     to_insert = []
#     to_update = []

#     for entry in price_history:
#         date = entry['date'].date() if isinstance(entry['date'], datetime) else entry['date']
#         if date in existing_dates:
#             to_update.append({
#                 'id': existing_dates[date],
#                 'asset_id': asset.id,
#                 'date': date,
#                 'open': entry['open'],
#                 'high': entry['high'],
#                 'low': entry['low'],
#                 'close': entry['close'],
#                 'volume': entry['volume']
#             })
#         else:
#             to_insert.append(PriceHistory(
#                 asset_id=asset.id,
#                 date=date,
#                 open=entry['open'],
#                 high=entry['high'],
#                 low=entry['low'],
#                 close=entry['close'],
#                 volume=entry['volume']
#             ))

#     if to_insert:
#         session.bulk_save_objects(to_insert)

#     if to_update:
#         session.bulk_update_mappings(PriceHistory, to_update)


# def get_all_assets(session: session_scope) -> List[Asset]:
#     return session.query(Asset).all()


# def get_all_currencies(session: session_scope) -> List[Asset]:
#     return session.query(Asset).filter(Asset.is_currency == True).all()


# def get_last_price_date(session: session_scope, ticker: str) -> Optional[datetime]:
#     asset = session.query(Asset).filter(Asset.ticker == ticker).first()
#     if asset:
#         last_price = session.query(PriceHistory).filter(PriceHistory.asset_id == asset.id).order_by(PriceHistory.date.desc()).first()
#         return last_price.date if last_price else None
#     return None


# def get_action_type_by_name(session: session_scope, name: str) -> Optional[ActionType]:
#     return session.query(ActionType).filter(ActionType.name == name).first()


# def get_asset_by_code(session: session_scope, code: str) -> Optional[Asset]:
#     return session.query(Asset).filter(Asset.code == code).first()


# def get_portfolio_by_name(session: session_scope, name: str) -> Optional[Portfolio]:
#     return session.query(Portfolio).filter(Portfolio.name == name).first()


# def get_unprocessed_actions(session: session_scope) -> List[Action]:
#     return session.query(Action).filter(Action.is_processed == False).order_by(Action.date).all()


# def update_portfolio_holdings_and_action(session: session_scope, action: Action):
#     quantity_old = (
#         session.query(PortfolioHolding.quantity_new)
#         .join(Action)
#         .filter(
#             (Action.portfolio_id == action.portfolio_id) &
#             (Action.asset_id == action.asset_id)
#         )
#         .order_by(PortfolioHolding.date.desc())
#         .first()
#     )

#     quantity_old = quantity_old[0] if quantity_old else 0

#     if action.action_type_id in (ActionTypeEnum.buy.value, ActionTypeEnum.dividend.value):
#         quantity_change = action.quantity
#     elif action.action_type_id == ActionTypeEnum.sell.value:
#         quantity_change = -action.quantity
#     else:
#         quantity_change = 0

#     session.add(PortfolioHolding(
#         action_id=action.asset_id,
#         portfolio_id=action.portfolio_id,
#         asset_id=action.asset_id,
#         quantity_change=quantity_change,
#         quantity_new=quantity_old+quantity_change,
#         date=action.date,
#         action=action
#     ))

#     session.query(Action).filter(Action.id == action.id).update({'is_processed': True})


# def get_last_holdings_time_series_update(session: session_scope) -> Optional[datetime]:
#     last_update = session.query(func.max(PortfolioHoldingsTimeSeries.date)).scalar()
#     return last_update.date() if last_update else None


# def clear_holdings_time_series(session: session_scope):
#     session.query(PortfolioHoldingsTimeSeries).delete()


# def get_earliest_action_date(session: session_scope) -> datetime:
#     earliest_date = session.query(func.min(Action.date)).scalar()
#     return earliest_date.date() if earliest_date else datetime.now().date()


# def get_all_portfolios(session: session_scope) -> List[Portfolio]:
#     return session.query(Portfolio).all()


# def get_portfolio_asset_actions(session: session_scope, portfolio_id: int, asset_id: int) -> List[Action]:
#     return session.query(Action).options(
#         joinedload(Action.action_type)
#     ).filter(
#         Action.portfolio_id == portfolio_id,
#         Action.asset_id == asset_id
#     ).order_by(Action.date).all()


# def merge_portfolio_holdings_time_series(session: session_scope, holdings: List[PortfolioHoldingsTimeSeries]):
#     for holding in holdings:
#         existing = session.query(PortfolioHoldingsTimeSeries).filter(
#             PortfolioHoldingsTimeSeries.portfolio_id == holding.portfolio_id,
#             PortfolioHoldingsTimeSeries.asset_id == holding.asset_id,
#             PortfolioHoldingsTimeSeries.date == holding.date
#         ).first()
#         if existing:
#             existing.quantity = holding.quantity
#         else:
#             session.add(holding)


# def get_portfolio_holdings_time_series(session: session_scope, portfolio_id: int, asset_id: int, start_date: datetime, end_date: datetime) -> List[PortfolioHoldingsTimeSeries]:
#     return session.query(PortfolioHoldingsTimeSeries).filter(
#         PortfolioHoldingsTimeSeries.portfolio_id == portfolio_id,
#         PortfolioHoldingsTimeSeries.asset_id == asset_id,
#         PortfolioHoldingsTimeSeries.date.between(start_date, end_date)
#     ).order_by(PortfolioHoldingsTimeSeries.date).all()