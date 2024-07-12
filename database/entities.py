
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey
from datetime import date
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass

class Asset(Base):
    __tablename__ = 'assets'
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str]

class PriceHistory(Base):
    __tablename__ = 'price_history'
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    date: Mapped[date]
    open: Mapped[float]
    high: Mapped[float]
    low: Mapped[float]
    close: Mapped[float]
    volume: Mapped[int]
    asset: Mapped["Asset"] = relationship()

class Portfolio(Base):
    __tablename__ = 'portfolios'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    owner: Mapped[str]

class PortfolioHolding(Base):
    __tablename__ = 'portfolio_holdings'
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    date: Mapped[date]
    quantity: Mapped[float]
    asset: Mapped["Asset"] = relationship()

class PortfolioMetric(Base):
    __tablename__ = 'portfolio_metrics'
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
    date: Mapped[date]
    total_value: Mapped[float]
    daily_return: Mapped[float]
    cumulative_return: Mapped[float]
    volatility: Mapped[float]
    sharpe_ratio: Mapped[float]
    portfolio: Mapped["Portfolio"] = relationship()
