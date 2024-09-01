
import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, and_, func, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, joinedload, Session
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import pandas as pd

from database.entities import *


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
    @wraps(func)
    def wrapper(*args, **kwargs):
        with session_scope() as session:
            if args and hasattr(args[0], func.__name__):
                # This is likely a method call, insert session after self
                return func(args[0], session, *args[1:], **kwargs)
            else:
                # This is likely a regular function call
                return func(session, *args, **kwargs)
    return wrapper


class DatabaseAccess:

    def __init__(self, engine: Optional[Engine] = None):
        if engine is None:
            self.engine = create_engine(os.environ.get('DATABASE_URL'))
        else:
            self.engine = engine


    def init_db(self) -> None:
        """Initialize the database by creating all tables."""
        Base.metadata.create_all(self.engine)

    def insert_if_not_exists(self, session: Session, model: Base, data: List[Dict], filter_fields: Optional[List[str]] = None) -> None:
        """
        Insert data into the database if it doesn't already exist.

        Args:
            session (Session): SQLAlchemy session
            model (Base): SQLAlchemy model class
            data (List[Dict]): List of dictionaries containing data to insert
            filter_fields (Optional[List[str]]): Fields to use for filtering existing data
        """
        for item in data:
            if filter_fields:
                filter_dict = {field: item[field] for field in filter_fields if field in item}
                exists = session.query(model).filter_by(**filter_dict).first()
            else: 
                exists = session.query(model).filter_by(**item).first()
            if not exists:
                obj = model(**item)
                session.add(obj)

    def add_asset(self, session: Session, ticker: str, name: str) -> Asset:
        """
        Add a new asset or update an existing one.

        Args:
            session (Session): SQLAlchemy session
            ticker (str): Asset ticker
            name (str): Asset name

        Returns:
            Asset: The added or updated asset
        """
        existing_asset = session.query(Asset).filter(Asset.ticker == ticker).first()

        if existing_asset:
            if existing_asset.name != name:
                existing_asset.name = name
            return existing_asset
        else:
            new_asset = Asset(ticker=ticker, name=name)
            session.add(new_asset)
            session.flush()
            return new_asset

    def add_price_history(self, session: Session, ticker: str, price_history: List[Dict]) -> None:
        """
        Add price history for an asset.

        Args:
            session (Session): SQLAlchemy session
            ticker (str): Asset ticker
            price_history (List[Dict]): List of price history entries
        """
        asset = session.query(Asset).filter(Asset.ticker == ticker).first()
        if not asset:
            raise ValueError(f"Asset with ticker {ticker} not found")

        min_date = min(entry['date'] for entry in price_history)

        existing_records = session.query(PriceHistory).filter(
            and_(
                PriceHistory.asset_id == asset.id,
                PriceHistory.date >= min_date
            )
        ).all()

        existing_dates = {record.date.date(): record.id for record in existing_records}

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

    def get_all_assets(self, session: Session) -> List[Asset]:
        """
        Get all assets.

        Args:
            session (Session): SQLAlchemy session

        Returns:
            List[Asset]: List of all assets
        """
        return session.query(Asset).all()

    def get_all_currencies(self, session: Session) -> List[Asset]:
        """
        Get all currency assets.

        Args:
            session (Session): SQLAlchemy session

        Returns:
            List[Asset]: List of all currency assets
        """
        return session.query(Asset).filter(Asset.is_currency == True).all()

    def get_last_price_date(self, session: Session, ticker: str) -> Optional[datetime]:
        """
        Get the date of the last price for a given asset.

        Args:
            session (Session): SQLAlchemy session
            ticker (str): Asset ticker

        Returns:
            Optional[datetime]: Date of the last price, or None if not found
        """
        asset = session.query(Asset).filter(Asset.ticker == ticker).first()
        if asset:
            last_price = session.query(PriceHistory).filter(PriceHistory.asset_id == asset.id).order_by(PriceHistory.date.desc()).first()
            return last_price.date if last_price else None
        return None

    def get_action_type_by_name(self, session: Session, name: str) -> Optional[ActionType]:
        """
        Get an action type by its name.

        Args:
            session (Session): SQLAlchemy session
            name (str): Name of the action type

        Returns:
            Optional[ActionType]: The action type, or None if not found
        """
        return session.query(ActionType).filter(ActionType.name == name).first()

    def get_asset_by_code(self, session: Session, code: str) -> Optional[Asset]:
        """
        Get an asset by its code.

        Args:
            session (Session): SQLAlchemy session
            code (str): Asset code

        Returns:
            Optional[Asset]: The asset, or None if not found
        """
        return session.query(Asset).filter(Asset.code == code).first()

    def get_portfolio_by_name(self, session: Session, name: str) -> Optional[Portfolio]:
        """
        Get a portfolio by its name.

        Args:
            session (Session): SQLAlchemy session
            name (str): Portfolio name

        Returns:
            Optional[Portfolio]: The portfolio, or None if not found
        """
        return session.query(Portfolio).filter(Portfolio.name == name).first()

    def get_unprocessed_actions(self, session: Session) -> List[Action]:
        """
        Get all unprocessed actions.

        Args:
            session (Session): SQLAlchemy session

        Returns:
            List[Action]: List of unprocessed actions
        """
        return session.query(Action).filter(Action.is_processed == False).order_by(Action.date).all()

    def update_portfolio_holdings_and_action(self, session: Session, action: Action) -> None:
        """
        Update portfolio holdings based on an action and mark the action as processed.

        Args:
            session (Session): SQLAlchemy session
            action (Action): The action to process
        """
        quantity_old = session.query(PortfolioHolding.quantity_new)\
            .join(Action)\
            .filter(
                (Action.portfolio_id == action.portfolio_id) &
                (Action.asset_id == action.asset_id)
            )\
            .order_by(PortfolioHolding.date.desc())\
            .first()

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

    def get_last_holdings_time_series_update(self, session: Session) -> Optional[datetime]:
        """
        Get the date of the last update to the holdings time series.

        Args:
            session (Session): SQLAlchemy session

        Returns:
            Optional[datetime]: Date of the last update, or None if no updates
        """
        last_update = session.query(func.max(PortfolioHoldingsTimeSeries.date)).scalar()
        return last_update.date() if last_update else None

    def clear_holdings_time_series(self, session: Session) -> None:
        """
        Clear all entries from the holdings time series.

        Args:
            session (Session): SQLAlchemy session
        """
        session.query(PortfolioHoldingsTimeSeries).delete()

    def get_earliest_action_date(self, session: Session) -> datetime:
        """
        Get the date of the earliest action.

        Args:
            session (Session): SQLAlchemy session

        Returns:
            datetime: Date of the earliest action, or current date if no actions
        """
        earliest_date = session.query(func.min(Action.date)).scalar()
        return earliest_date.date() if earliest_date else datetime.now().date()

    def get_portfolios(self, session: Session) -> List[Portfolio]:
        """
        Get all portfolios.

        Args:
            session (Session): SQLAlchemy session

        Returns:
            List[Portfolio]: List of all portfolios
        """
        return session.query(Portfolio).all()

    def get_portfolio_assets(self, session: Session, portfolio_id: int) -> List[Asset]:
        """
        Get all unique assets associated with a specific portfolio.

        This method retrieves all assets that have been involved in any action
        within the specified portfolio.

        Args:
            session (Session): SQLAlchemy session
            portfolio_id (int): ID of the portfolio

        Returns:
            List[Asset]: A list of unique Asset objects associated with the portfolio
        """
        # TODO: this is wrong, no join on Action
        return session.query(Asset).join(Action).filter(Action.portfolio_id == portfolio_id).distinct().all()

    def get_portfolio_asset_actions(self, session: Session, portfolio_id: int, asset_id: int) -> List[Action]:
        """
        Get all actions for a specific asset in a specific portfolio.

        Args:
            session (Session): SQLAlchemy session
            portfolio_id (int): ID of the portfolio
            asset_id (int): ID of the asset

        Returns:
            List[Action]: List of actions
        """
        return session.query(Action).options(
            joinedload(Action.action_type)
        ).filter(
            Action.portfolio_id == portfolio_id,
            Action.asset_id == asset_id
        ).order_by(Action.date).all()

    def store_holdings_time_series(self, session: Session, holdings_data: List[Dict]) -> None:
        """
        Store or update portfolio holdings time series data.

        Args:
            session (Session): SQLAlchemy session
            holdings_data (List[Dict]): List of dictionaries containing holdings data to store or update

        Returns:
            None
        """
        for holding in holdings_data:
            session.merge(PortfolioHoldingsTimeSeries(**holding))

    # def store_holdings_time_series(self, session: Session, holdings: List[PortfolioHoldingsTimeSeries]) -> None:
    #     """
    #     Merge new holdings into the time series, updating existing entries or adding new ones.

    #     Args:
    #         session (Session): SQLAlchemy session
    #         holdings (List[PortfolioHoldingsTimeSeries]): List of holdings to merge
    #     """
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

    def get_portfolio_holdings_time_series(self, session: Session, portfolio_id: int, asset_id: int, start_date: datetime, end_date: datetime) -> List[PortfolioHoldingsTimeSeries]:
        """
        Get the holdings time series for a specific asset in a specific portfolio within a date range.

        Args:
            session (Session): SQLAlchemy session
            portfolio_id (int): ID of the portfolio
            asset_id (int): ID of the asset
            start_date (datetime): Start date of the range
            end_date (datetime): End date of the range

        Returns:
            List[PortfolioHoldingsTimeSeries]: List of holdings time series entries
        """
        return session.query(PortfolioHoldingsTimeSeries).filter(
            PortfolioHoldingsTimeSeries.portfolio_id == portfolio_id,
            PortfolioHoldingsTimeSeries.asset_id == asset_id,
            PortfolioHoldingsTimeSeries.date.between(start_date, end_date)
        ).order_by(PortfolioHoldingsTimeSeries.date).all()

    def update_holding_time_series(self, session: Session, action: Action, end_date: datetime) -> None:
        """
        Forward fill the holding time series for a specific action up to the end date.

        Args:
            session (Session): SQLAlchemy session
            action (Action): The action to forward fill from
            end_date (datetime): The end date to fill up to

        Returns:
            None
        """

        # Get the latest holding entry for this action
        latest_holding = session.query(PortfolioHoldingsTimeSeries).filter(
            PortfolioHoldingsTimeSeries.portfolio_id == action.portfolio_id,
            PortfolioHoldingsTimeSeries.asset_id == action.asset_id,
            PortfolioHoldingsTimeSeries.date <= action.date
        ).order_by(PortfolioHoldingsTimeSeries.date.desc()).first()

        # If no previous holding exists, use the action's quantity
        if latest_holding:
            quantity = latest_holding.quantity 
        else:
            if action.action_type.name in ('buy', 'dividend'):
                quantity = action.quantity
            elif action.action_type.name == 'sell':
                quantity = -action.quantity

        # Generate date range from the day after the action to the end date
        start_date = (action.date + timedelta(days=1)).date()
        while start_date <= end_date:
            # Check if an entry already exists for this date
            existing_entry = session.query(PortfolioHoldingsTimeSeries).filter(
                PortfolioHoldingsTimeSeries.portfolio_id == action.portfolio_id,
                PortfolioHoldingsTimeSeries.asset_id == action.asset_id,
                PortfolioHoldingsTimeSeries.date == start_date
            ).first()

            if existing_entry:
                # If an entry exists, update its quantity
                existing_entry.quantity = quantity
            else:
                # If no entry exists, create a new one
                new_entry = PortfolioHoldingsTimeSeries(
                    portfolio_id=action.portfolio_id,
                    asset_id=action.asset_id,
                    date=start_date,
                    quantity=quantity
                )
                session.add(new_entry)

            start_date += timedelta(days=1)


    def insert_holding_time_series_ffill(self, session: Session, action: Action, end_date: datetime) -> None:
        """
        Update the portfolio holdings time series using forward fill (ffill) method.

        This method creates or updates daily holdings entries for a specific asset in a portfolio
        from the day after the last known holding up to the specified end date.
        It uses the most recent known quantity and forward-fills it for all subsequent dates.

        Args:
            session (Session): The database session for executing queries and updates.
            action (Action): The action object containing portfolio_id, asset_id, and date information.
            end_date (datetime): The end date up to which the holdings should be updated.

        Returns:
            None
        """
        # Get the latest holding entry for this action
        latest_holding = session.query(PortfolioHoldingsTimeSeries).filter(
            PortfolioHoldingsTimeSeries.portfolio_id == action.portfolio_id,
            PortfolioHoldingsTimeSeries.asset_id == action.asset_id,
        ).order_by(PortfolioHoldingsTimeSeries.date.desc()).first()

        # If no previous holding exists, use the action's quantity
        if latest_holding:
            start_date = (latest_holding.date).date()
            date_range = pd.date_range(start=start_date, end=end_date)
            if start_date < end_date and len(date_range) > 1:
                quantity = latest_holding.quantity 
                # Create a DataFrame with all dates in the range
                df = pd.DataFrame({
                    'date': date_range,
                    'portfolio_id': action.portfolio_id,
                    'asset_id': action.asset_id,
                    'quantity': quantity
                })
                records = df.to_dict('records')
                stmt = insert(PortfolioHoldingsTimeSeries).values(records)
                try:
                    session.execute(stmt)
                except Exception as e:
                    logging.error(f"Error ffill holdings time series: {str(e)}")
                    logging.error(f"Action details: portfolio_id={action.portfolio_id}, asset_id={action.asset_id}, date={action.date}")
                    raise e


    def update_holding_time_series_vectorized(self, session: Session, action: Action, end_date: datetime) -> None:
        """
        Forward fill the holding time series for a specific action up to the end date.

        Args:
            session (Session): SQLAlchemy session
            action (Action): The action to forward fill from
            end_date (datetime): The end date to fill up to

        Returns:
            None
        """

        # ---------------
        # This section performs a ffill to the current date
        # ---------------

        if action.action_type.name in ('buy', 'dividend'):
            quantity_change = action.quantity
        elif action.action_type.name == 'sell':
            quantity_change = -action.quantity

        # Generate date range
        start_date = action.date.date()

        # Update existing records
        update_stmt = (
            update(PortfolioHoldingsTimeSeries)
            .where(
                PortfolioHoldingsTimeSeries.portfolio_id == action.portfolio_id,
                PortfolioHoldingsTimeSeries.asset_id == action.asset_id,
                PortfolioHoldingsTimeSeries.date >= start_date,
                PortfolioHoldingsTimeSeries.date <= end_date
            )
            .values(quantity=PortfolioHoldingsTimeSeries.quantity + quantity_change)
        )
        
        try:
            session.execute(update_stmt)
        except Exception as e:
            logging.error(f"Error updating holdings time series: {str(e)}")
            logging.error(f"Action details: portfolio_id={action.portfolio_id}, asset_id={action.asset_id}, date={action.date}")
            raise e


        # date_range = pd.date_range(start=start_date, end=end_date)

        # # Create a DataFrame with all dates in the range
        # df = pd.DataFrame({
        #     'date': date_range,
        #     'portfolio_id': action.portfolio_id,
        #     'asset_id': action.asset_id
        #     # 'quantity': quantity
        # })

        # # Query existing holdings directly into a DataFrame
        # query = session.query(PortfolioHoldingsTimeSeries).filter(
        #     PortfolioHoldingsTimeSeries.portfolio_id == action.portfolio_id,
        #     PortfolioHoldingsTimeSeries.asset_id == action.asset_id,
        #     PortfolioHoldingsTimeSeries.date >= start_date,
        #     PortfolioHoldingsTimeSeries.date <= end_date
        #     )
        # existing_df = pd.read_sql(query.statement, session.bind, parse_dates=['date'])

        # if not existing_df.empty:
        #     # If holdings exist, merge with df and update quantities
        #     df = df.merge(existing_df[['date', 'quantity']], on='date', how='left')
        #     df['quantity'] = df['quantity'].fillna(0)  # Fill NaNs with 0
        #     df['quantity'] += quantity_change
        # else:
        #     # If no holdings exist, set quantity to the action quantity
        #     df['quantity'] = quantity_change

        # # Convert to list of dicts for bulk insert/update
        # records = df.to_dict('records')

        # # This needs to be an update
        # stmt = insert(PortfolioHoldingsTimeSeries).values(records)

        # try:
        #     session.execute(stmt)
        # except Exception as e:
        #     logging.error(f"Error inserting/updating holdings time series: {str(e)}")
        #     logging.error(f"Action details: portfolio_id={action.portfolio_id}, asset_id={action.asset_id}, date={action.date}")
        #     raise e

if __name__ == "__main__":

    db_access = DatabaseAccess()	
    with session_scope() as session:
        portfolio = db_access.get_portfolio_by_name(session, 'Alexander')

    print('done')