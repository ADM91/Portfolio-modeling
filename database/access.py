
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import pandas as pd

from database.entities import Base, Asset, PriceHistory, PortfolioHolding, Portfolio, PortfolioMetric


class DatabaseAccess:
    def __init__(self, db_url='sqlite:///data/asset_tracker.db'):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.init_db()  # creates tabes if they don't exist

    def init_db(self):
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self):
        with Session(self.engine) as session:
            try:
                yield session
                session.commit()
            except:
                session.rollback()
                raise

    def add_asset(self, ticker, name):
        with self.session_scope() as session:
            asset = Asset(ticker=ticker, name=name)
            session.add(asset)

    def add_price_history(self, ticker, prices_df):
        with self.session_scope() as session:
            asset = session.query(Asset).filter_by(ticker=ticker).first()
            for _, row in prices_df.iterrows():
                price_history = PriceHistory(asset_id=asset.id, date=row.name, price=row['price'])
                session.add(price_history)

    def get_price_history(self, ticker, start_date=None, end_date=None):
        with self.session_scope() as session:
            query = session.query(PriceHistory).join(Asset).filter(Asset.ticker == ticker)
            if start_date:
                query = query.filter(PriceHistory.date >= start_date)
            if end_date:
                query = query.filter(PriceHistory.date <= end_date)
            result = query.order_by(PriceHistory.date).all()
            return pd.DataFrame([(ph.date, ph.price) for ph in result], columns=['date', 'price']).set_index('date')

    def update_portfolio_holdings(self, portfolio_id, holdings_df):
        with self.session_scope() as session:
            for _, row in holdings_df.iterrows():
                holding = PortfolioHolding(portfolio_id=portfolio_id, asset_id=row['asset_id'], 
                                           date=row['date'], quantity=row['quantity'])
                session.add(holding)

    def update_portfolio_metrics(self, portfolio_id, metrics_df):
        with self.session_scope() as session:
            for date, row in metrics_df.iterrows():
                metric = PortfolioMetric(portfolio_id=portfolio_id, date=date, 
                                         total_value=row['total_value'],
                                         daily_return=row['daily_return'],
                                         cumulative_return=row['cumulative_return'],
                                         volatility=row['volatility'],
                                         sharpe_ratio=row['sharpe_ratio'])
                session.add(metric)

    def get_portfolio_holdings(self, portfolio_id, start_date=None, end_date=None):
        with self.session_scope() as session:
            query = session.query(PortfolioHolding).filter_by(portfolio_id=portfolio_id)
            if start_date:
                query = query.filter(PortfolioHolding.date >= start_date)
            if end_date:
                query = query.filter(PortfolioHolding.date <= end_date)
            result = query.order_by(PortfolioHolding.date).all()
            return pd.DataFrame([(ph.date, ph.asset_id, ph.quantity) for ph in result], 
                                columns=['date', 'asset_id', 'quantity'])

    def get_portfolio_metrics(self, portfolio_id, start_date=None, end_date=None):
        with self.session_scope() as session:
            query = session.query(PortfolioMetric).filter_by(portfolio_id=portfolio_id)
            if start_date:
                query = query.filter(PortfolioMetric.date >= start_date)
            if end_date:
                query = query.filter(PortfolioMetric.date <= end_date)
            result = query.order_by(PortfolioMetric.date).all()
            return pd.DataFrame([(pm.date, pm.total_value, pm.daily_return, pm.cumulative_return, 
                                  pm.volatility, pm.sharpe_ratio) for pm in result],
                                columns=['date', 'total_value', 'daily_return', 'cumulative_return', 
                                         'volatility', 'sharpe_ratio']).set_index('date')

    def get_asset_prices(self, asset_ids, start_date=None, end_date=None):
        with self.session_scope() as session:
            query = session.query(PriceHistory).filter(PriceHistory.asset_id.in_(asset_ids))
            if start_date:
                query = query.filter(PriceHistory.date >= start_date)
            if end_date:
                query = query.filter(PriceHistory.date <= end_date)
            result = query.order_by(PriceHistory.date).all()
            return pd.DataFrame([(ph.asset_id, ph.date, ph.price) for ph in result],
                                columns=['asset_id', 'date', 'price'])
        

if __name__ == "__main__":

    DatabaseAccess()