
from enum import Enum
from typing import Optional
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey, DateTime, Boolean, Float, Integer


class Base(DeclarativeBase):
    pass

class ActionTypeEnum(Enum):
    buy = 1
    sell = 2
    dividend = 3

class ActionType(Base):
    # reference table
    __tablename__ = 'action_types'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))

class Action(Base):
    # investment action, buy, sell, dividend (currency agnostic)
    __tablename__ = 'actions'
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
    action_type_id: Mapped[int] = mapped_column(ForeignKey('action_types.id'))
    date: Mapped[datetime] = mapped_column(DateTime)
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    currency_id: Mapped[int] = mapped_column(ForeignKey('assets.id')) # how do we handle dividend payments? currency = asset
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    fee: Mapped[float] = mapped_column(Float)
    platform: Mapped[Optional[str]] = mapped_column(String)
    comment: Mapped[Optional[str]] = mapped_column(String)
    is_processed: Mapped[bool] = mapped_column(Boolean) 
    action_type: Mapped[ActionType] = relationship("ActionType")
    asset: Mapped["Asset"] = relationship("Asset", foreign_keys=[asset_id])
    currency: Mapped["Asset"] = relationship("Asset", foreign_keys=[currency_id])
    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="actions")

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

class Asset(Base):
    # reference table
    __tablename__ = 'assets'
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(unique=True)
    code: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String(50))
    is_currency: Mapped[bool] = mapped_column(Boolean)
    is_inverted: Mapped[bool] = mapped_column(Boolean)
    price_history: Mapped[list["PriceHistory"]] = relationship("PriceHistory", back_populates="asset", foreign_keys=[PriceHistory.asset_id])
    actions: Mapped[list["Action"]] = relationship("Action", back_populates="asset", foreign_keys=[Action.asset_id])
    currency_actions: Mapped[list["Action"]] = relationship("Action", back_populates="currency", foreign_keys=[Action.currency_id])
    holdings_time_series: Mapped[list["PortfolioHoldingsTimeSeries"]] = relationship("PortfolioHoldingsTimeSeries", back_populates="asset")

class PortfolioHoldingsTimeSeries(Base):
    __tablename__ = 'portfolio_holdings_time_series'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(Integer, ForeignKey('portfolios.id'))
    asset_id: Mapped[int] = mapped_column(Integer, ForeignKey('assets.id'))
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="holdings_time_series")
    asset: Mapped["Asset"] = relationship("Asset", back_populates="holdings_time_series")

class Portfolio(Base):
    __tablename__ = 'portfolios'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    owner: Mapped[str] = mapped_column(String)
    actions: Mapped[list["Action"]] = relationship("Action", back_populates="portfolio", foreign_keys=[Action.portfolio_id])
    holdings_time_series: Mapped[list["PortfolioHoldingsTimeSeries"]] = relationship("PortfolioHoldingsTimeSeries", back_populates="portfolio")


# TODO: consider adding a receipts table to keep track of all purchase receipts for tax reasons


# TODO:  remove this table, replaced by PortfolioHoldingsTimeSeries
# class PortfolioHolding(Base):
#     __tablename__ = 'portfolio_holdings'
#     id: Mapped[int] = mapped_column(primary_key=True)
#     action_id: Mapped[int] = mapped_column(ForeignKey('actions.id'))
#     portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
#     asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
#     quantity_change: Mapped[float] = mapped_column(Float)
#     quantity_new: Mapped[float] = mapped_column(Float)
#     date: Mapped[datetime] = mapped_column(DateTime)
#     action: Mapped["Action"] = relationship("Action")
#     portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="holdings")
#     asset: Mapped["Asset"] = relationship("Asset", back_populates="portfolio_holdings")

# class MetricType(Base):
#     # base currency agnostic
#     __tablename__ = 'metric_types'
#     id: Mapped[int] = mapped_column(primary_key=True)
#     name: Mapped[str] = mapped_column(String(50))

# class Metric(Base):
#     # base currency agnostic, time series
#     __tablename__ = 'metrics'
#     id: Mapped[int] = mapped_column(primary_key=True)
#     metric_type_id: Mapped[int] = mapped_column(ForeignKey('metric_types.id'))
#     portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
#     holding_id: Mapped[Optional[int]] = mapped_column(ForeignKey('portfolio_holdings.id'), nullable=True)  # Null for portfolio-level metrics
#     currency_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
#     date: Mapped[datetime] = mapped_column(DateTime)
#     value: Mapped[float] = mapped_column(Float)
#     metric_type: Mapped[MetricType] = relationship("MetricType")
#     portfolio: Mapped[Portfolio] = relationship("Portfolio", back_populates="metrics")
#     holding: Mapped[Optional[PortfolioHolding]] = relationship("PortfolioHolding", back_populates="metrics")
#     currency: Mapped[Asset] = relationship("Asset", back_populates="metrics")
