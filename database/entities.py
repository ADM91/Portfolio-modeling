
from typing import Optional
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey, DateTime


class Base(DeclarativeBase):
    pass

class ActionType(Base):
    # reference table
    __tablename__ = 'action_types'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))

class Action(Base):
    # investment action, buy, sell, dividend (currency agnostic)
    __tablename__ = 'actions'
    id: Mapped[int] = mapped_column(primary_key=True)
    action_type_id: Mapped[int] = mapped_column(ForeignKey('action_types.id'))
    name: Mapped[str] = mapped_column(String(50))
    implemented: Mapped[bool] = mapped_column()
    action_type: Mapped[ActionType] = relationship("ActionType")

class Currency(Base):
    # refernce table
    __tablename__ = 'currencies'
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(unique=True)
    code: Mapped[str] = mapped_column(String(3), unique=True)
    name: Mapped[str] = mapped_column(String(50))
    exchange_rates: Mapped[list["ExchangeRate"]] = relationship("ExchangeRate", back_populates="currency")

class ExchangeRate(Base):
    # USD rate, time series
    __tablename__ = 'exchange_rates'
    id: Mapped[int] = mapped_column(primary_key=True)
    currency_id: Mapped[int] = mapped_column(ForeignKey('currencies.id'))
    date: Mapped[datetime] = mapped_column(DateTime)
    rate: Mapped[float] = mapped_column()  # rate to USD
    currency: Mapped[Currency] = relationship("Currency", back_populates="exchange_rates")

class Asset(Base):
    # refernce table
    __tablename__ = 'assets'
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(unique=True)
    code: Mapped[str] = mapped_column()
    name: Mapped[str] = mapped_column(String(50))
    holdings: Mapped[list["PortfolioHolding"]] = relationship("PortfolioHolding", back_populates="asset")
    price_history: Mapped[list["PriceHistory"]] = relationship("PriceHistory", back_populates="asset")

class PriceHistory(Base):
    # usd rate, time series
    __tablename__ = 'price_history'
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    date: Mapped[datetime] = mapped_column(DateTime)
    open: Mapped[float] = mapped_column()
    high: Mapped[float] = mapped_column()
    low: Mapped[float] = mapped_column()
    close: Mapped[float] = mapped_column()
    volume: Mapped[int] = mapped_column()
    asset: Mapped["Asset"] = relationship("Asset", back_populates="price_history")

class Portfolio(Base):
    # refernce table
    __tablename__ = 'portfolios'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()
    owner: Mapped[str] = mapped_column()
    holdings: Mapped[list["PortfolioHolding"]] = relationship("PortfolioHolding", back_populates="portfolio")
    metrics: Mapped[list["Metric"]] = relationship("Metric", back_populates="portfolio")

class PortfolioHolding(Base):
    # in kind, time series
    __tablename__ = 'portfolio_holdings'
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    quantity: Mapped[float] = mapped_column()
    date: Mapped[datetime] = mapped_column(DateTime)
    portfolio: Mapped[Portfolio] = relationship("Portfolio", back_populates="holdings")
    asset: Mapped[Asset] = relationship("Asset", back_populates="holdings")
    metrics: Mapped[list["Metric"]] = relationship("Metric", back_populates="holding")

class MetricType(Base):
    # base currency agnostic
    __tablename__ = 'metric_types'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    
class Metric(Base):
    # base currency agnostic, time series
    __tablename__ = 'metrics'
    id: Mapped[int] = mapped_column(primary_key=True)
    metric_type_id: Mapped[int] = mapped_column(ForeignKey('metric_types.id'))
    portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
    holding_id: Mapped[Optional[int]] = mapped_column(ForeignKey('portfolio_holdings.id'), nullable=True)  # Null for portfolio-level metrics
    currency_id: Mapped[int] = mapped_column(ForeignKey('currencies.id'))
    date: Mapped[datetime] = mapped_column(DateTime)
    value: Mapped[float] = mapped_column()
    metric_type: Mapped[MetricType] = relationship("MetricType")
    portfolio: Mapped[Portfolio] = relationship("Portfolio", back_populates="metrics")
    holding: Mapped[Optional[PortfolioHolding]] = relationship("PortfolioHolding", back_populates="metrics")
    currency: Mapped[Currency] = relationship("Currency")

# Backpopulate relationships
Currency.exchange_rates = relationship("ExchangeRate", back_populates="currency")
Asset.price_history = relationship("PriceHistory", back_populates="asset")
Asset.holdings = relationship("PortfolioHolding", back_populates="asset")
Portfolio.holdings = relationship("PortfolioHolding", back_populates="portfolio")
Portfolio.metrics = relationship("Metric", back_populates="portfolio")
PortfolioHolding.metrics = relationship("Metric", back_populates="holding")
