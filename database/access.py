
import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, and_, func, update, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, joinedload, Session
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta, date
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

    def get_currency_conversion_on_date(self, session: Session, from_currency_id: int, to_currency_id: int, date: datetime):

        # Price on last available date
        from_currency_price = session.query(PriceHistory.date, PriceHistory.close.label('from_currency_price'))\
            .filter(PriceHistory.asset_id == from_currency_id)\
            .filter(PriceHistory.date <= date)\
            .order_by(PriceHistory.date.desc())\
            .limit(1)\
            .subquery()
        
        to_currency_price = session.query(PriceHistory.date, PriceHistory.close.label('to_currency_price'))\
            .filter(PriceHistory.asset_id == to_currency_id)\
            .filter(PriceHistory.date <= date)\
            .order_by(PriceHistory.date.desc())\
            .limit(1)\
            .subquery()

        # Perform the division
        exchange_rate = session.query(
            (from_currency_price.c.from_currency_price / to_currency_price.c.to_currency_price).label('exchange_rate')
        ).first()

        # The result is now available as exchange_rate.exchange_rate
        return exchange_rate.exchange_rate

    def get_currency_conversion_time_series(self, session: Session, from_currency_id: int, to_currency_id: int, start_date: datetime, end_date: datetime):

        # Fetch price history for both currencies
        from_currency_prices = session.query(PriceHistory.date, PriceHistory.close.label('from_currency_price'))\
            .join(Asset, PriceHistory.asset_id == Asset.id)\
            .filter(and_(PriceHistory.asset_id == from_currency_id,
                        PriceHistory.date >= start_date,
                        PriceHistory.date <= end_date))\
            .subquery()

        to_currency_prices = session.query(PriceHistory.date, PriceHistory.close.label('to_currency_price'))\
            .join(Asset, PriceHistory.asset_id == Asset.id)\
            .filter(and_(PriceHistory.asset_id == to_currency_id,
                        PriceHistory.date >= start_date,
                        PriceHistory.date <= end_date))\
            .subquery()

        # Join the price histories
        conversion_rates = session.query(
            from_currency_prices.c.date,
            from_currency_prices.c.from_currency_price,
            to_currency_prices.c.to_currency_price,
        ).join(to_currency_prices, from_currency_prices.c.date == to_currency_prices.c.date)

        return conversion_rates

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

    def get_buy_sell_actions_by_portfolio_id_asset_id(self, session: Session, portfolio_id: int, asset_id: int) -> Session.query:

        actions = (
            session.query(Action, ActionType.name.label('action_type_name'))
            .join(ActionType, Action.action_type_id == ActionType.id)
            .filter(
                Action.portfolio_id == portfolio_id,
                Action.asset_id == asset_id,
                Action.action_type_id.in_([1, 2])  # 1 for buy, 2 for sell
            )
        )
        
        return actions

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

    def update_action(self, session: Session, action: Action) -> None:
        """
        Update an action to mark it as processed.
        
        This method updates the 'is_processed' field of the given action to True.

        Args:
            session (Session): The SQLAlchemy session for database operations.
            action (Action): The action object to be updated.

        Returns:
            None

        Note:
            This method only updates the 'is_processed' field and does not commit the session.
            The caller is responsible for committing or rolling back the session as needed.
        """
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

    def get_asset_price_history(self, session: Session, asset_id: int, start_date: datetime, end_date: datetime) -> List[PriceHistory]:
        """
        Get asset prices for a specific asset between the specified date range.

        Args:
            session (Session): SQLAlchemy session
            asset_id (int): ID of the asset
            start_date (datetime): Start date of the range
            end_date (datetime): End date of the range

        Returns:
            List[PriceHistory]: A list of PriceHistory objects for the specified asset and date range,
                                ordered by date
        """
        return session.query(PriceHistory).filter(
            PriceHistory.asset_id == asset_id,
            PriceHistory.date.between(start_date, end_date)
        ).order_by(PriceHistory.date).all()

    def get_asset_price_history_df(self, session: Session, asset_id: int, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get asset prices for a specific asset between the specified date range as a pandas DataFrame.

        Args:
            session (Session): SQLAlchemy session
            asset_id (int): ID of the asset
            start_date (datetime): Start date of the range
            end_date (datetime): End date of the range

        Returns:
            pd.DataFrame: A DataFrame containing PriceHistory data for the specified asset and date range,
                                ordered by date
        """

        # Construct the SQL query
        query = select(PriceHistory.asset_id, PriceHistory.date, PriceHistory.close).filter(
            PriceHistory.asset_id == asset_id,
            PriceHistory.date >= start_date,
            PriceHistory.date < end_date + timedelta(days=1)  # ensure that we don´t miss the last day
        ).order_by(PriceHistory.date)

        # Execute the query and fetch results directly into a DataFrame
        df = pd.read_sql(query, session.bind)

        return df

    def get_portfolio_asset_time_series(self, session: Session, portfolio_id: int, asset_id: int) -> List[PortfolioHoldingsTimeSeries]:
        """
        Get the complete time series of holdings for a specific asset in a specific portfolio.

        Args:
            session (Session): SQLAlchemy session
            portfolio_id (int): ID of the portfolio
            asset_id (int): ID of the asset

        Returns:
            List[PortfolioHoldingsTimeSeries]: List of all holdings time series entries for the specified asset and portfolio
        """
        return session.query(PortfolioHoldingsTimeSeries).filter(
            PortfolioHoldingsTimeSeries.portfolio_id == portfolio_id,
            PortfolioHoldingsTimeSeries.asset_id == asset_id
        ).order_by(PortfolioHoldingsTimeSeries.date).all()

    def get_portfolio_asset_time_series_df(self, session: Session, portfolio_id: int, asset_id: int) -> pd.DataFrame:
        """
        Get the complete time series of holdings for a specific asset in a specific portfolio as a pandas DataFrame.

        Args:
            session (Session): SQLAlchemy session
            portfolio_id (int): ID of the portfolio
            asset_id (int): ID of the asset

        Returns:
            pd.DataFrame: DataFrame containing all holdings time series entries for the specified asset and portfolio
        """
        # Construct the SQL query
        query = select(PortfolioHoldingsTimeSeries).filter(
            PortfolioHoldingsTimeSeries.portfolio_id == portfolio_id,
            PortfolioHoldingsTimeSeries.asset_id == asset_id
        ).order_by(PortfolioHoldingsTimeSeries.date)

        # Execute the query and fetch results directly into a DataFrame
        df = pd.read_sql(query, session.bind)

        return df

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
        Update the portfolio holdings time series for a specific action up to the end date using a vectorized approach.

        This method updates or inserts holdings entries for a specific asset in a portfolio
        from the action date up to the specified end date. It calculates the quantity change
        based on the action type and applies this change to all relevant dates.
        Args:
            session (Session): The SQLAlchemy session for database operations.
            action (Action): The action object containing portfolio_id, asset_id, date, and quantity information.
            end_date (datetime): The end date up to which the holdings should be updated.

        Returns:
            None

        Raises:
            Exception: If there's an error during the update or insert process.

        Note:
            This method uses a vectorized approach for better performance when dealing with large datasets.
            It first attempts to update existing records and then inserts new records if necessary.
        """

        if action.action_type.name in ('buy', 'dividend'):
            quantity_change = action.quantity
        elif action.action_type.name == 'sell':
            quantity_change = -action.quantity

        start_date = action.date.date()
        # end_date = end_date.date()

        try:
            # First, update existing records
            update_stmt = (
                update(PortfolioHoldingsTimeSeries)
                .where(
                    and_(
                        PortfolioHoldingsTimeSeries.portfolio_id == action.portfolio_id,
                        PortfolioHoldingsTimeSeries.asset_id == action.asset_id,
                        PortfolioHoldingsTimeSeries.date >= start_date,
                        PortfolioHoldingsTimeSeries.date <= end_date + timedelta(days=1) 
                    )
                )
                .values(quantity=PortfolioHoldingsTimeSeries.quantity + quantity_change)
            )
            result = session.execute(update_stmt)
            
            # If no rows were updated, we need to insert new records
            if result.rowcount == 0:
                # Generate date series in Python
                date_series = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

                # Prepare data for insertion
                insert_data = [
                    {
                        'portfolio_id': action.portfolio_id,
                        'asset_id': action.asset_id,
                        'date': date,
                        'quantity': quantity_change
                    }
                    for date in date_series
                ]

                # Insert new records
                insert_stmt = insert(PortfolioHoldingsTimeSeries)
                session.execute(insert_stmt, insert_data)

        except Exception as e:
            session.rollback()
            logging.error(f"Error updating/inserting holdings time series: {str(e)}")
            logging.error(f"Action details: portfolio_id={action.portfolio_id}, asset_id={action.asset_id}, date={action.date}")
            raise e

    def update_holdings_time_series_to_current_day(self, session: Session) -> None:
        """
        Update the portfolio holdings time series for all portfolios and assets up to the current day.

        This method finds the last date in the holdings time series for each portfolio-asset combination
        and fills in any missing dates up to the current day, using the last known quantity.

        Args:
            session (Session): SQLAlchemy session

        Returns:
            None
        """
        current_date = datetime.now().date()

        # Get the latest date for each portfolio-asset combination
        latest_dates = session.query(
            PortfolioHoldingsTimeSeries.portfolio_id,
            PortfolioHoldingsTimeSeries.asset_id,
            func.max(PortfolioHoldingsTimeSeries.date).label('last_date')
        ).group_by(
            PortfolioHoldingsTimeSeries.portfolio_id,
            PortfolioHoldingsTimeSeries.asset_id
        ).subquery()

        # Join with the main table to get the latest quantity
        latest_holdings = session.query(
            PortfolioHoldingsTimeSeries
        ).join(
            latest_dates,
            and_(
                PortfolioHoldingsTimeSeries.portfolio_id == latest_dates.c.portfolio_id,
                PortfolioHoldingsTimeSeries.asset_id == latest_dates.c.asset_id,
                PortfolioHoldingsTimeSeries.date == latest_dates.c.last_date
            )
        ).all()

        # Update or insert new entries
        for holding in latest_holdings:
            start_date = (holding.date + timedelta(days=1)).date()
            if start_date <= current_date:
                date_range = pd.date_range(start=start_date, end=current_date)
                new_entries = [
                    PortfolioHoldingsTimeSeries(
                        portfolio_id=holding.portfolio_id,
                        asset_id=holding.asset_id,
                        date=date,
                        quantity=holding.quantity
                    )
                    for date in date_range
                ]
                session.bulk_save_objects(new_entries)

        session.commit()


if __name__ == "__main__":

    db_access = DatabaseAccess()	
    with session_scope() as session:
        portfolio = db_access.get_portfolio_by_name(session, 'Alexander')

    print('done')