from enum import Enum
from typing import Optional
from datetime import datetime
from decimal import Decimal
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey, DateTime, Boolean, Float, Integer, Numeric, Index, UniqueConstraint, CheckConstraint


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
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    currency_id: Mapped[int] = mapped_column(ForeignKey('assets.id')) # how do we handle dividend payments? currency = asset
    
    # Use Decimal for financial precision
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    fee: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=0)
    
    platform: Mapped[Optional[str]] = mapped_column(String(100))
    comment: Mapped[Optional[str]] = mapped_column(String(500))
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False) 
    
    # Relationships
    action_type: Mapped[ActionType] = relationship("ActionType")
    asset: Mapped["Asset"] = relationship("Asset", foreign_keys=[asset_id])
    currency: Mapped["Asset"] = relationship("Asset", foreign_keys=[currency_id])
    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="actions")
    tax_lot_transactions: Mapped[list["TaxLotTransaction"]] = relationship("TaxLotTransaction", back_populates="action")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_action_portfolio_date', 'portfolio_id', 'date'),
        Index('idx_action_asset_date', 'asset_id', 'date'), 
        Index('idx_action_processed', 'is_processed'),
    )

class PriceHistory(Base):
    # price history in original asset denomination
    __tablename__ = 'price_history'
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Use Decimal for price precision
    open: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[Optional[int]] = mapped_column(Integer)
    
    asset: Mapped["Asset"] = relationship("Asset", back_populates="price_history")
    
    __table_args__ = (
        UniqueConstraint('asset_id', 'date', name='uq_price_history_asset_date'),
        Index('idx_price_history_asset_date', 'asset_id', 'date'),
    )

class Asset(Base):
    # reference table
    __tablename__ = 'assets'
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), unique=True)
    code: Mapped[str] = mapped_column(String(20))
    name: Mapped[str] = mapped_column(String(100))
    is_currency: Mapped[bool] = mapped_column(Boolean, default=False)
    is_inverted: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    price_history: Mapped[list["PriceHistory"]] = relationship("PriceHistory", back_populates="asset", foreign_keys=[PriceHistory.asset_id])
    actions: Mapped[list["Action"]] = relationship("Action", back_populates="asset", foreign_keys=[Action.asset_id])
    currency_actions: Mapped[list["Action"]] = relationship("Action", back_populates="currency", foreign_keys=[Action.currency_id])
    holdings_time_series: Mapped[list["PortfolioHoldingsTimeSeries"]] = relationship("PortfolioHoldingsTimeSeries", back_populates="asset")
    tax_lots: Mapped[list["TaxLot"]] = relationship("TaxLot", back_populates="asset")
    
    __table_args__ = (
        Index('idx_asset_ticker', 'ticker'),
        Index('idx_asset_is_currency', 'is_currency'),
    )

class PortfolioHoldingsTimeSeries(Base):
    __tablename__ = 'portfolio_holdings_time_series'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(Integer, ForeignKey('portfolios.id'))
    asset_id: Mapped[int] = mapped_column(Integer, ForeignKey('assets.id'))
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    
    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="holdings_time_series")
    asset: Mapped["Asset"] = relationship("Asset", back_populates="holdings_time_series")
    
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'asset_id', 'date', name='uq_holdings_time_series'),
        Index('idx_holdings_portfolio_date', 'portfolio_id', 'date'),
        Index('idx_holdings_asset_date', 'asset_id', 'date'),
    )

class Portfolio(Base):
    __tablename__ = 'portfolios'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    owner: Mapped[str] = mapped_column(String(100))
    
    actions: Mapped[list["Action"]] = relationship("Action", back_populates="portfolio", foreign_keys=[Action.portfolio_id])
    holdings_time_series: Mapped[list["PortfolioHoldingsTimeSeries"]] = relationship("PortfolioHoldingsTimeSeries", back_populates="portfolio")
    tax_lots: Mapped[list["TaxLot"]] = relationship("TaxLot", back_populates="portfolio")

# ========== Tax Lot Tables (New) ==========

class TaxLot(Base):
    """Tax lot tracking for cost basis calculations - essential for tax reporting"""
    __tablename__ = 'tax_lots'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'), nullable=False)
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'), nullable=False)
    
    # Lot identification
    lot_identifier: Mapped[str] = mapped_column(String(100), nullable=False)  # Unique identifier for the lot
    acquisition_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Quantity and cost basis
    original_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    remaining_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    cost_basis_per_unit: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    
    # Currency for cost basis
    currency_id: Mapped[int] = mapped_column(ForeignKey('assets.id'), nullable=False)
    
    # Status
    is_closed: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="tax_lots")
    asset: Mapped["Asset"] = relationship("Asset", back_populates="tax_lots")
    currency: Mapped["Asset"] = relationship("Asset", foreign_keys=[currency_id])
    transactions: Mapped[list["TaxLotTransaction"]] = relationship("TaxLotTransaction", back_populates="tax_lot")
    
    __table_args__ = (
        Index('idx_tax_lot_portfolio_asset', 'portfolio_id', 'asset_id'),
        Index('idx_tax_lot_acquisition_date', 'acquisition_date'),
        Index('idx_tax_lot_is_closed', 'is_closed'),
        UniqueConstraint('portfolio_id', 'asset_id', 'lot_identifier', name='uq_tax_lot_identifier'),
        CheckConstraint('remaining_quantity >= 0', name='check_remaining_quantity_non_negative'),
        CheckConstraint('remaining_quantity <= original_quantity', name='check_remaining_lte_original'),
    )

class TaxLotTransaction(Base):
    """Links actions to tax lot changes for tracking cost basis"""
    __tablename__ = 'tax_lot_transactions'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    action_id: Mapped[int] = mapped_column(ForeignKey('actions.id'), nullable=False)
    tax_lot_id: Mapped[int] = mapped_column(ForeignKey('tax_lots.id'), nullable=False)
    
    # Transaction details
    quantity_change: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    
    # For sales - realized gain/loss tracking
    proceeds: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    realized_gain_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    
    # Processing timestamp
    processed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    action: Mapped["Action"] = relationship("Action", back_populates="tax_lot_transactions")
    tax_lot: Mapped["TaxLot"] = relationship("TaxLot", back_populates="transactions")
    
    __table_args__ = (
        Index('idx_tax_lot_transaction_action', 'action_id'),
        Index('idx_tax_lot_transaction_lot', 'tax_lot_id'),
    )
