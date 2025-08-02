"""
Custom exceptions for the database access layer.
Provides specific error types for better error handling and debugging.
"""

from typing import Optional, Any


class DatabaseError(Exception):
    """Base exception for database-related errors"""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""
    pass


class DatabaseOperationError(DatabaseError):
    """Raised when a database operation fails"""
    
    def __init__(self, operation: str, message: str, details: Optional[dict] = None):
        super().__init__(f"Database operation '{operation}' failed: {message}", details)
        self.operation = operation


class AssetNotFoundError(DatabaseError):
    """Raised when an asset is not found"""
    
    def __init__(self, identifier: str, identifier_type: str = "ticker"):
        message = f"Asset not found with {identifier_type}: {identifier}"
        super().__init__(message, {"identifier": identifier, "identifier_type": identifier_type})
        self.identifier = identifier
        self.identifier_type = identifier_type


class PortfolioNotFoundError(DatabaseError):
    """Raised when a portfolio is not found"""
    
    def __init__(self, identifier: str, identifier_type: str = "name"):
        message = f"Portfolio not found with {identifier_type}: {identifier}"
        super().__init__(message, {"identifier": identifier, "identifier_type": identifier_type})
        self.identifier = identifier
        self.identifier_type = identifier_type


class CurrencyConversionError(DatabaseError):
    """Raised when currency conversion fails"""
    
    def __init__(self, from_currency: str, to_currency: str, date: str, reason: str):
        message = f"Currency conversion failed from {from_currency} to {to_currency} on {date}: {reason}"
        super().__init__(message, {
            "from_currency": from_currency,
            "to_currency": to_currency,
            "date": date,
            "reason": reason
        })


class DataValidationError(DatabaseError):
    """Raised when data validation fails"""
    
    def __init__(self, field: str, value: Any, reason: str):
        message = f"Data validation failed for field '{field}' with value '{value}': {reason}"
        super().__init__(message, {"field": field, "value": value, "reason": reason})
        self.field = field
        self.value = value


class TaxLotError(DatabaseError):
    """Raised when tax lot operations fail"""
    
    def __init__(self, operation: str, message: str, lot_id: Optional[int] = None):
        super().__init__(f"Tax lot {operation} failed: {message}", {"lot_id": lot_id})
        self.operation = operation
        self.lot_id = lot_id


class PriceDataError(DatabaseError):
    """Raised when price data operations fail"""
    
    def __init__(self, ticker: str, date: str, operation: str, reason: str):
        message = f"Price data {operation} failed for {ticker} on {date}: {reason}"
        super().__init__(message, {
            "ticker": ticker,
            "date": date,
            "operation": operation,
            "reason": reason
        })
