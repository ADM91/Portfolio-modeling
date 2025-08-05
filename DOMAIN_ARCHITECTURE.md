# Domain Architecture with Instructive File Naming

This document explains the new domain architecture with clear, instructive file naming conventions that make the purpose of each file immediately obvious.

## 📁 File Structure Overview

```
domain/
├── __init__.py                 # Clean imports for the domain layer
├── entities.py                 # Core business entities (Portfolio, Asset, ActionType)
├── investment_action.py        # Investment action model with business logic
└── portfolio_holding.py        # Portfolio holding model with business operations
```

## 🎯 File Naming Convention

The file names are designed to be **immediately descriptive** of their contents:

### `entities.py` - Core Business Entities
- **What it contains**: Simple data structures representing core business concepts
- **Examples**: `Portfolio`, `Asset`, `ActionType`
- **Purpose**: These are the fundamental building blocks of the business domain
- **Why this name**: "Entities" clearly indicates these are the core business objects

### `investment_action.py` - Investment Action Domain Model
- **What it contains**: The `InvestmentAction` class with all its business logic
- **Examples**: Validation, calculations, business rules
- **Purpose**: Represents the core business concept of an investment transaction
- **Why this name**: The file name exactly matches the main class it contains

### `portfolio_holding.py` - Portfolio Holding Domain Model
- **What it contains**: The `PortfolioHolding` class with business operations
- **Examples**: Static methods for batch processing, validation, calculations
- **Purpose**: Represents portfolio holdings and provides complex business operations
- **Why this name**: The file name exactly matches the main class it contains

## 🔍 Benefits of This Structure

### 1. **Immediate Clarity**
- File names tell you exactly what's inside
- No need to open files to understand their purpose
- New developers can navigate the codebase intuitively

### 2. **Logical Organization**
- Related functionality is grouped together
- Each file has a single, clear responsibility
- Easy to find where to add new features

### 3. **Scalable Structure**
- Easy to add new domain models as separate files
- Each business concept gets its own dedicated file
- No monolithic files that become hard to maintain

### 4. **Clean Imports**
- The `__init__.py` provides clean imports: `from domain import InvestmentAction`
- Internal structure is hidden from consumers
- Easy to refactor internal organization without breaking imports

## 📋 What Each File Contains

### `domain/entities.py`
```python
# Simple, focused entities
class ActionType(Enum):
    BUY = "buy"
    SELL = "sell" 
    DIVIDEND = "dividend"

@dataclass
class Asset:
    id: int
    ticker: str
    code: str
    name: str
    # ... other simple properties

@dataclass  
class Portfolio:
    id: int
    name: str
    owner: str
```

### `domain/investment_action.py`
```python
# Rich domain model with business logic
class InvestmentAction:
    def __init__(self, ...):
        # Construction and validation
        
    def calculate_total_cost(self) -> Decimal:
        # Business calculations
        
    def is_long_term_holding(self) -> bool:
        # Business rules
        
    def _validate(self) -> None:
        # Business validation
```

### `domain/portfolio_holding.py`
```python
# Domain model with complex business operations
class PortfolioHolding:
    def apply_action(self, action) -> 'PortfolioHolding':
        # Business operations
        
    @staticmethod
    def validate_action_sequence(actions) -> List[str]:
        # Complex validation logic
        
    @staticmethod
    def process_action_batch(actions) -> Tuple[List, List[str]]:
        # Complex business operations
```

## 🚀 Usage Examples

### Clean Imports
```python
# Import everything you need from the domain
from domain import InvestmentAction, Portfolio, Asset, ActionType, PortfolioHolding

# Create business objects
portfolio = Portfolio(id=1, name="My Portfolio", owner="John")
asset = Asset(id=1, ticker="AAPL", code="AAPL", name="Apple Inc.")

# Use business logic
action = InvestmentAction(
    portfolio=portfolio,
    action_type=ActionType.BUY,
    # ... other parameters
)

# Business calculations
total_cost = action.calculate_total_cost()
is_long_term = action.is_long_term_holding()

# Complex operations
errors = PortfolioHolding.validate_action_sequence([action])
processed, errors = PortfolioHolding.process_action_batch([action])
```

### Adding New Domain Models
When you need to add a new domain concept, create a new file:

```
domain/
├── entities.py
├── investment_action.py
├── portfolio_holding.py
└── tax_calculation.py        # New domain model
```

Then update `domain/__init__.py`:
```python
from .tax_calculation import TaxCalculation

__all__ = [
    # ... existing exports
    'TaxCalculation',
]
```

## 🎯 Design Principles

### 1. **One Class Per File** (for complex models)
- Each significant domain model gets its own file
- File name matches the main class name
- Makes it easy to find and maintain code

### 2. **Grouped Simple Entities**
- Simple entities are grouped in `entities.py`
- These are typically dataclasses or simple classes
- Reduces file proliferation for simple concepts

### 3. **Business Logic Co-location**
- All logic related to a domain concept is in its file
- No hunting across multiple files for related functionality
- Easy to understand the complete behavior of a domain object

### 4. **Clear Separation of Concerns**
- `entities.py`: Simple data structures
- `investment_action.py`: Individual action logic
- `portfolio_holding.py`: Portfolio-level operations and batch processing

## 🔄 Migration Benefits

### Before (Monolithic)
```
domain/
└── models.py    # Everything mixed together - 500+ lines
```

### After (Organized)
```
domain/
├── entities.py           # ~50 lines - Simple entities
├── investment_action.py  # ~150 lines - Action logic
└── portfolio_holding.py  # ~200 lines - Portfolio operations
```

### Results
- ✅ **Easier to navigate**: Find what you need quickly
- ✅ **Easier to maintain**: Changes are localized
- ✅ **Easier to test**: Each file can be tested independently
- ✅ **Easier to understand**: Clear separation of concerns
- ✅ **Easier to extend**: Add new models without affecting existing ones

## 🎉 Conclusion

This file structure makes the domain layer **immediately understandable** and **highly maintainable**:

- **File names are descriptive** - you know what's inside without opening them
- **Logical organization** - related functionality is grouped together  
- **Scalable structure** - easy to add new domain models
- **Clean imports** - simple, consistent import patterns
- **Single responsibility** - each file has one clear purpose

The architecture supports both current needs and future growth while maintaining clarity and simplicity.
