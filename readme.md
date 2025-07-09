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
#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
#### 2. Initialize the database:
```bash
python startup.py startup
```
    
## TODO
 - ~~read and insert actions needs to be able to interpret USD as currency~~
 - ~~action service to read excel data, clean/interpret, insert/update database~~
 - ~~portfolio service updates portfolio and portfolioholdings tables~~
 - ~~change DatabaseAccess methods to return session queries, not detached data objects.  Like done "in get_portfolio_asset_actions()". Session lifecycle moves to business logic layer.~~
 - ~~working on update_holding_time_series() in DatabaseAccess~~
 - ~~time series for holdings in kind - separate table or dataframe in ram. - separate table~~
 - ~~transform in-kind holdings to currency denominated holdings test performing in ram with pandas vs sql querying~~
 - ~~simple plotting for proof of concept~~
 - ~~want to be able to view metrics for combinations of portfolios and assets~~
- ~~backend services as fastapi~~
 - adding core metric calcualtion functions - ~~value invested~~ ~~cost basis~~, ~~unrealized gain/loss~~, ~~time-weighted return~~, ~~sharpe ratio~~, realized gain/loss, get all
 - summary statistics for metrics - for each portfolio, total value, total invested, total unrealized gain/loss, ytd return, 1 year return, 30 day return, sharpe ratio
 - tax calculations, FIFO method (USA), cost basis method (Iceland)
 - add enpoints for all data outputs
 - frontend
 - dividends robust
 - action defined on precise datetime (currently data aggregates on day, we lose fidelity of inter day transactions)

test