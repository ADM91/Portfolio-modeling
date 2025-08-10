from enum import Enum
from typing import Optional
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey, DateTime, Boolean, Float, Integer, Numeric, Index, UniqueConstraint, CheckConstraint


class Base(DeclarativeBase):
    pass

class TransactionTypeEnum(Enum):
    buy = 1
    sell = 2
    dividend = 3

class TransactionType(Base):
    # reference table
    __tablename__ = 'transaction_types'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))

class Transaction(Base):
    # investment action, buy, sell, dividend (currency agnostic)
    __tablename__ = 'transactions'
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
    action_type_id: Mapped[int] = mapped_column(ForeignKey('transaction_types.id'))
    transaction_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    currency_id: Mapped[int] = mapped_column(ForeignKey('assets.id')) # how do we handle dividend payments? currency = asset
    
    # Use Decimal for financial precision
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    fee: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=0)
    
    # Metadata
    platform: Mapped[Optional[str]] = mapped_column(String(100))
    comment: Mapped[Optional[str]] = mapped_column(String(500))
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False) 
    
    # Relationships
    transaction_type: Mapped[TransactionType] = relationship("TransactionType")
    asset: Mapped["Asset"] = relationship("Asset", foreign_keys=[asset_id])
    currency: Mapped["Asset"] = relationship("Asset", foreign_keys=[currency_id])
    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="transactions")
    tax_lot_transactions: Mapped[list["TaxLotTransaction"]] = relationship("TaxLotTransaction", back_populates="transaction")

    # Indexes for performance
    __table_args__ = (
        Index('idx_transaction_portfolio_date', 'portfolio_id', 'transaction_datetime'),
        Index('idx_transaction_asset_date', 'asset_id', 'transaction_datetime'), 
        Index('idx_transaction_processed', 'is_processed'),
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
    transactions: Mapped[list["Transaction"]] = relationship("Transaction", back_populates="asset", foreign_keys=[Transaction.asset_id])
    currency_transactions: Mapped[list["Transaction"]] = relationship("Transaction", back_populates="currency", foreign_keys=[Transaction.currency_id])
    holdings_time_series: Mapped[list["PortfolioHoldingsTimeSeries"]] = relationship("PortfolioHoldingsTimeSeries", back_populates="asset")
    tax_lots: Mapped[list["TaxLot"]] = relationship("TaxLot", back_populates="asset", foreign_keys="TaxLot.asset_id")
    
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

    transactions: Mapped[list["Transaction"]] = relationship("Transaction", back_populates="portfolio", foreign_keys=[Transaction.portfolio_id])
    holdings_time_series: Mapped[list["PortfolioHoldingsTimeSeries"]] = relationship("PortfolioHoldingsTimeSeries", back_populates="portfolio")
    tax_lots: Mapped[list["TaxLot"]] = relationship("TaxLot", back_populates="portfolio")

# ========== Tax Lot Tables ==========

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
    
    # For US tax reporting
    holding_period_type: Mapped[Optional[str]] = mapped_column(String(10))  # 'short' or 'long'
    acquisition_method: Mapped[str] = mapped_column(String(20), default='purchase')  # 'purchase', 'dividend_reinvest', etc.

    # For wash sale tracking
    is_wash_sale_affected: Mapped[bool] = mapped_column(Boolean, default=False)
    wash_sale_loss_deferred: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))

    # Currency for cost basis
    currency_id: Mapped[int] = mapped_column(ForeignKey('assets.id'), nullable=False)
    
    # Status
    is_closed: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="tax_lots")
    asset: Mapped["Asset"] = relationship("Asset", back_populates="tax_lots", foreign_keys=[asset_id])
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
    """Links transactions to tax lot changes for tracking cost basis"""
    __tablename__ = 'tax_lot_transactions'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey('transactions.id'), nullable=False)
    tax_lot_id: Mapped[int] = mapped_column(ForeignKey('tax_lots.id'), nullable=False)
    
    # Transaction details
    quantity_change: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    
    # For sales - realized gain/loss tracking
    proceeds: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    realized_gain_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    
    # Processing timestamp
    processed_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    transaction: Mapped["Transaction"] = relationship("Transaction", back_populates="tax_lot_transactions")
    tax_lot: Mapped["TaxLot"] = relationship("TaxLot", back_populates="transactions")
    
    __table_args__ = (
        Index('idx_tax_lot_transaction_transaction', 'transaction_id'),
        Index('idx_tax_lot_transaction_lot', 'tax_lot_id'),
    )


class WashSaleEvent(Base):
    __tablename__ = 'wash_sale_events'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    
    sale_transaction_id: Mapped[int] = mapped_column(ForeignKey('transactions.id'))
    purchase_transaction_id: Mapped[int] = mapped_column(ForeignKey('transactions.id'))

    disallowed_loss: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    wash_sale_period_start: Mapped[datetime] = mapped_column(DateTime)
    wash_sale_period_end: Mapped[datetime] = mapped_column(DateTime)

# ========== Log Tables ==========

class AuditLog(Base):
    __tablename__ = 'audit_log'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    table_name: Mapped[str] = mapped_column(String(50), nullable=False)
    record_id: Mapped[int] = mapped_column(Integer, nullable=False)
    operation: Mapped[str] = mapped_column(String(10), nullable=False)  # 'INSERT', 'UPDATE', 'DELETE'
    
    old_values: Mapped[Optional[str]] = mapped_column(String(2000))  # JSON
    new_values: Mapped[Optional[str]] = mapped_column(String(2000))  # JSON
    
    changed_by: Mapped[str] = mapped_column(String(100), default='system')
    changed_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
