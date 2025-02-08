# Portfolio Modeling

A comprehensive multi-currency portfolio tracking and analytics system built with FastAPI, SQLAlchemy, and Streamlit.

## Features
- 📊 Multi-currency portfolio tracking
- 📈 Financial metrics calculation (Sharpe ratio, time-weighted returns, etc.)
- 🔄 Automatic market data updates via Yahoo Finance
- 🌐 RESTful API with FastAPI
- 📱 Web interface with Streamlit
- 💾 SQLite database with SQLAlchemy ORM
## Structure
- **config.py**: Configuration settings with environment variable support
- **startup.py**: CLI tool for database initialization and server management
- **main.py**: FastAPI application entry point
- **requirements.txt**: Project dependencies
- **database/**: 
  - `access.py`: Database access and operations
  - `entities.py`: Database entities/models defined by SQLAlchemy ORM
  - `init_db.py`: Database initialization script
- **services/**:
  - `action_service.py`: Manages trading actions
  - `metric_service.py`: Calculates financial metrics
  - `portfolio_service.py`: Manages portfolios
  - `yfinance_service.py`: Interacts with Yahoo Finance API
- **routers/**: FastAPI route handlers
  - `action_router.py`: Action-related endpoints
  - `metric_router.py`: Metrics and analytics endpoints
  - `portfolio_router.py`: Portfolio management endpoints
- **frontend/**: Streamlit web interface
- **utils/**: Utility functions

## Quick Start
### 1. Install Dependencies
```bash
    pip install -r requirements.txt
```

