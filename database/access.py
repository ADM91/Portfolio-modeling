
from contextlib import contextmanager
from datetime import datetime, date
from typing import List, Dict

import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session

from database.entities import *


class DatabaseAccess:
    def __init__(self, db_url='sqlite:///database/asset_tracker.db'):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def init_db(self):
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self):
        session = self.Session()
        try:
            yield session
            session.commit()
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

            # Fetch all existing dates for this asset
            existing_dates = set(date[0] for date in session.query(PriceHistory.date)
                                .filter(PriceHistory.asset_id == asset.id).all())

            # Prepare bulk insert and update lists
            to_insert = []
            to_update = []

            for entry in price_history:
                date = entry['date'].date() if isinstance(entry['date'], datetime) else entry['date']
                if date in existing_dates:
                    to_update.append({
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
    
    def get_all_currencies(self) -> List[Currency]:
        with self.session_scope() as session:
            currencies = session.query(Currency).all()
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
        
    # def get_asset(self, ticker: str):
    #     with self.session_scope() as session:
    #         return session.execute(select(Asset).where(Asset.ticker == ticker)).scalar_one_or_none()

    # def get_price_history(self, ticker: str, start_date: date = None, end_date: date = None):
    #     with self.session_scope() as session:
    #         stmt = (
    #             select(PriceHistory)
    #             .join(Asset)
    #             .where(Asset.ticker == ticker)
    #             .order_by(PriceHistory.date)
    #         )
    #         if start_date:
    #             stmt = stmt.where(PriceHistory.date >= start_date)
    #         if end_date:
    #             stmt = stmt.where(PriceHistory.date <= end_date)
            
    #         result = session.execute(stmt).scalars().all()
    #         return pd.DataFrame([
    #             {
    #                 'date': ph.date,
    #                 'open': ph.open,
    #                 'high': ph.high,
    #                 'low': ph.low,
    #                 'close': ph.close,
    #                 'volume': ph.volume
    #             } for ph in result
    #         ]).set_index('date')

    # def fetch_and_store_yfinance_data(self, ticker: str, start_date: datetime = None, end_date: datetime = None):
    #     if not start_date:
    #         start_date = datetime.now() - pd.Timedelta(days=5*365)  # 5 years ago
    #     if not end_date:
    #         end_date = datetime.now()

    #     yf_ticker = yf.Ticker(ticker)
        
    #     # Fetch asset info
    #     info = yf_ticker.info
    #     asset_name = info.get('longName', info.get('shortName', ticker))
    #     self.add_asset(ticker, asset_name)

    #     # Fetch price data
    #     price_data = yf_ticker.history(start=start_date, end=end_date)
    #     self.add_price_history(ticker, price_data)

    #     print(f"Added data for {ticker}")

    # def get_transformed_price_history(self, ticker: str, start_date: date = None, end_date: date = None):
    #     df = self.get_price_history(ticker, start_date, end_date)
        
    #     # Calculate 20-day moving average
    #     df['MA20'] = df['close'].rolling(window=20).mean()

    #     # Calculate daily returns
    #     df['daily_return'] = df['close'].pct_change()

    #     return df

# Usage example
if __name__ == "__main__":
    db_access = DatabaseAccess()
    db_access.init_db()

    # Fetch and store data for multiple tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    for ticker in tickers:
        db_access.fetch_and_store_yfinance_data(ticker)

    # Retrieve and transform data for AAPL
    aapl_data = db_access.get_transformed_price_history('AAPL')
    print(aapl_data.tail())