
import os
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict

from sqlalchemy import create_engine, select, and_
from sqlalchemy.orm import sessionmaker

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
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def insert_if_not_exists(self, model, data: List[Dict]):
        with self.session_scope() as session:
            for item in data:
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
            actions = session.query(Action).filter(Action.is_processed == False).all()
            # Expunge all objects from the session
            session.expunge_all()
            return actions



# Usage example
if __name__ == "__main__":
    db_access = DatabaseAccess()
    db_access.init_db()
