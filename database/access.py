
import os
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Optional

from sqlalchemy import create_engine, select, and_, func
from sqlalchemy.orm import sessionmaker, contains_eager, joinedload

from database.entities import *


class DatabaseAccess:
    def __init__(self):
        self.engine = create_engine(os.environ.get('DATABASE_URL'))
        self.Session = sessionmaker(bind=self.engine)

    def init_db(self):
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self):
        session = self.Session()
        try:
            yield session
            session.commit()  # runs after "with" block finishes
        except Exception as e:
            session.rollback()
            logging.error(f"Transaction failed: {e}")
            raise
        finally:
            session.close()

    def insert_if_not_exists(self, model, data: List[Dict], filter_fields: Optional[List[str]] = None):
        with self.session_scope() as session:
            for item in data:
                if filter_fields:
                    filter_dict = {field: item[field] for field in filter_fields if field in item}
                    exists = session.query(model).filter_by(**filter_dict).first()
                else: 
                    exists = session.query(model).filter_by(**item).first()
                if not exists:
                    obj = model(**item)
                    session.add(obj)

    def add_asset(self, ticker: str, name: str):
        with self.session_scope() as session:
            # Check if the asset already exists
            existing_asset = session.execute(
                select(Asset).where(Asset.ticker == ticker)
            ).scalar_one_or_none()

            if existing_asset:
                # If the asset exists, update the name if it's different
                if existing_asset.name != name:
                    existing_asset.name = name
                return existing_asset
            else:
                # If the asset doesn't exist, create a new one
                new_asset = Asset(ticker=ticker, name=name)
                session.add(new_asset)
                session.flush()  # This will populate the id of the new asset
                return new_asset

    def add_price_history(self, ticker: str, price_history: List[Dict]):
        with self.session_scope() as session:
            # Get the asset within this session
            asset = session.execute(select(Asset).where(Asset.ticker == ticker)).scalar_one_or_none()
            if not asset:
                raise ValueError(f"Asset with ticker {ticker} not found")

            # Get the minimum date from the input price history
            min_date = min(entry['date'] for entry in price_history)

            # Fetch existing records within the date range
            existing_records = session.execute(
                select(PriceHistory)
                .where(
                    and_(
                        PriceHistory.asset_id == asset.id,
                        PriceHistory.date >= min_date
                        )
                )
            ).all()

            # Get existing dates and ids
            existing_dates = {record[0].date.date(): record[0].id for record in existing_records}

            # Prepare bulk insert and update lists
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

            # Bulk insert new entries
            if to_insert:
                session.bulk_save_objects(to_insert)

            # Bulk update existing entries
            if to_update:
                session.bulk_update_mappings(PriceHistory, to_update)

    def get_all_assets(self) -> List[Asset]:
        with self.session_scope() as session:
            assets = session.query(Asset).all()
            # Expunge all objects from the session
            session.expunge_all()
        return assets
    
    def get_all_currencies(self) -> List[Asset]:
        with self.session_scope() as session:
            currencies = session.query(Asset).filter(Asset.is_currency == True).all()
            # Expunge all objects from the session
            session.expunge_all()
        return currencies

    def get_last_price_date(self, ticker: str) -> Optional[datetime]:
        with self.session_scope() as session:
            asset = session.query(Asset).filter(Asset.ticker == ticker).first()
            if asset:
                last_price = session.query(PriceHistory).filter(PriceHistory.asset_id == asset.id).order_by(PriceHistory.date.desc()).first()
                return last_price.date if last_price else None
            return None
        
    def get_action_type_by_name(self, name: str) -> Optional[ActionType]:
        with self.session_scope() as session:
            action_type = session.query(ActionType).filter(ActionType.name == name).first()
            # Expunge all objects from the session
            session.expunge_all()
            return action_type
        
    def get_asset_by_code(self, code: str) -> Optional[Asset]:
        with self.session_scope() as session:
            asset = session.query(Asset).filter(Asset.code == code).first()
            # Expunge all objects from the session
            session.expunge_all()
            return asset
        
    def get_portfolio_by_name(self, name: str) -> Optional[Portfolio]:
        with self.session_scope() as session:
            portfolio = session.query(Portfolio).filter(Portfolio.name == name).first()
            # Expunge all objects from the session
            session.expunge_all()
            return portfolio

    def get_unprocessed_actions(self) -> List[Action]:
        with self.session_scope() as session:
            actions = session.query(Action).filter(Action.is_processed == False).order_by(Action.date).all()
            # Expunge all objects from the session
            session.expunge_all()
            return actions
        
    def update_portfolio_holdings_and_action(self, action: Action):
        with self.session_scope() as session:
            
            # Get last quantity
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

            # Get the float value
            quantity_old = quantity_old[0] if quantity_old else 0

            # Add or subtract from quantity_old
            if action.action_type_id in (ActionTypeEnum.buy.value, ActionTypeEnum.dividend.value):
                quantity_change = action.quantity
            elif action.action_type_id == ActionTypeEnum.sell.value:
                quantity_change = -action.quantity
            else:
                quantity_change = 0

            # TODO: figure out why this does not work here, but works after i add the new entry to PortfolioHolding
            # if action.action_type.name in ('buy', 'dividend'):
            #     quantity_change = action.quantity
            # elif action.action_type.name == 'sell':
            #     quantity_change = -action.quantity
            # else:
            #     quantity_change = 0

            # Add the new entry to PortfolioHolding
            session.add(PortfolioHolding(
                action_id=action.asset_id,
                portfolio_id=action.portfolio_id,
                asset_id=action.asset_id,
                quantity_change=quantity_change,
                quantity_new=quantity_old+quantity_change,
                date=action.date,
                action=action
            ))

            # Update action as processed
            session.query(Action).filter(Action.id == action.id).update({'is_processed': True})

        return

    def get_last_holdings_time_series_update(self) -> Optional[datetime]:
        with self.session_scope() as session:
            last_update = session.query(func.max(PortfolioHoldingsTimeSeries.date)).scalar()
            return last_update.date() if last_update else None

    def clear_holdings_time_series(self):
        with self.session_scope() as session:
            session.query(PortfolioHoldingsTimeSeries).delete()

    def get_earliest_action_date(self) -> datetime:
        with self.session_scope() as session:
            earliest_date = session.query(func.min(Action.date)).scalar()
            return earliest_date.date() if earliest_date else datetime.now().date()

    def get_all_portfolios(self) -> List[Portfolio]:
        with self.session_scope() as session:
            portfolios = session.query(Portfolio).all()
            session.expunge_all()
        return portfolios

    # def get_portfolio_asset_actions(self, portfolio_id: int, asset_id: int) -> List[Action]:
    #     with self.session_scope() as session:
    #         actions = session.query(Action).filter(
    #             Action.portfolio_id == portfolio_id,
    #             Action.asset_id == asset_id
    #         ).order_by(Action.date).all()
    #         session.expunge_all()
    #     return actions

    # def get_portfolio_asset_actions(self, portfolio_id: int, asset_id: int) -> List[Action]:
    #     with self.session_scope() as session:
    #         actions = session.query(Action).options(
    #             contains_eager(Action.action_type)
    #         ).join(
    #             Action.action_type
    #         ).filter(
    #             Action.portfolio_id == portfolio_id,
    #             Action.asset_id == asset_id
    #         ).order_by(Action.date).all()
            
    #         # Detach the objects from the session
    #         for action in actions:
    #             session.expunge(action)
    #             session.expunge(action.action_type)
    #     return actions

    def get_portfolio_asset_actions(self, session, portfolio_id: int, asset_id: int) -> List[Action]:
        return session.query(Action).options(
            joinedload(Action.action_type)
        ).filter(
            Action.portfolio_id == portfolio_id,
            Action.asset_id == asset_id
        ).order_by(Action.date).all()


    def merge_portfolio_holdings_time_series(self, holdings: List[PortfolioHoldingsTimeSeries]):
        with self.session_scope() as session:
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

    def get_portfolio_holdings_time_series(self, portfolio_id: int, asset_id: int, start_date: datetime, end_date: datetime) -> List[PortfolioHoldingsTimeSeries]:
        with self.session_scope() as session:
            holdings = session.query(PortfolioHoldingsTimeSeries).filter(
                PortfolioHoldingsTimeSeries.portfolio_id == portfolio_id,
                PortfolioHoldingsTimeSeries.asset_id == asset_id,
                PortfolioHoldingsTimeSeries.date.between(start_date, end_date)
            ).order_by(PortfolioHoldingsTimeSeries.date).all()
            session.expunge_all()
        return holdings

# Usage example
if __name__ == "__main__":
    db_access = DatabaseAccess()
    db_access.init_db()
