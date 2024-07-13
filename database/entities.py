from typing import Optional
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey, DateTime, Boolean, Float, Integer

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
    implemented: Mapped[bool] = mapped_column(Boolean)
    action_type: Mapped[ActionType] = relationship("ActionType")

class Asset(Base):
    # reference table
    __tablename__ = 'assets'
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(unique=True)
    code: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String(50))
    is_currency: Mapped[bool] = mapped_column(Boolean)
    is_inverted: Mapped[bool] = mapped_column(Boolean)
    price_history: Mapped[list["PriceHistory"]] = relationship("PriceHistory", back_populates="asset")
    holdings: Mapped[list["PortfolioHolding"]] = relationship("PortfolioHolding", back_populates="asset")
    metrics: Mapped[list["Metric"]] = relationship("Metric", back_populates="currency")

class PriceHistory(Base):
    # usd rate, time series
    __tablename__ = 'price_history'
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    date: Mapped[datetime] = mapped_column(DateTime)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)
    asset: Mapped["Asset"] = relationship("Asset", back_populates="price_history")

class Portfolio(Base):
    # reference table
    __tablename__ = 'portfolios'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    owner: Mapped[str] = mapped_column(String)
    holdings: Mapped[list["PortfolioHolding"]] = relationship("PortfolioHolding", back_populates="portfolio")
    metrics: Mapped[list["Metric"]] = relationship("Metric", back_populates="portfolio")

class PortfolioHolding(Base):
    # in kind, time series
    __tablename__ = 'portfolio_holdings'
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    quantity: Mapped[float] = mapped_column(Float)
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
    currency_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    date: Mapped[datetime] = mapped_column(DateTime)
    value: Mapped[float] = mapped_column(Float)
    metric_type: Mapped[MetricType] = relationship("MetricType")
    portfolio: Mapped[Portfolio] = relationship("Portfolio", back_populates="metrics")
    holding: Mapped[Optional[PortfolioHolding]] = relationship("PortfolioHolding", back_populates="metrics")
    currency: Mapped[Asset] = relationship("Asset", back_populates="metrics")

# Backpopulate relationships
Asset.price_history = relationship("PriceHistory", back_populates="asset")
Asset.holdings = relationship("PortfolioHolding", back_populates="asset")
Asset.metrics = relationship("Metric", back_populates="currency")
Portfolio.holdings = relationship("PortfolioHolding", back_populates="portfolio")
Portfolio.metrics = relationship("Metric", back_populates="portfolio")
PortfolioHolding.metrics = relationship("Metric", back_populates="holding")
