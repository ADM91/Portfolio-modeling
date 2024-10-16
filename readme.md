
# Portfolio modeling

## Structure
- **config.py**: Configuration settings.
- **requirements.txt**: Project dependencies.
- **database/**: 
  - `access.py`: Database access and operations.
  - `entities.py`: Database entities/models defined by SQLAlchemy ORM.
  - `init_db.py`: Database initialization script.
- **services/**: 
  - `action_service.py`: Manages actions.
  - `metric_service.py`: Manages metrics.
  - `portfolio_service.py`: Manages portfolios.
  - `yfinance_service.py`: Interacts with Yahoo Finance API.

## Setup
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Initialize the database:
    ```bash
    python main.py
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
 - adding core metric calcualtion functions - ~~value invested~~ cost basis, unrealized gain/loss, realized gain/loss, time-weighted return, sharpe ratio
 - dividends robust
 - backend services as fastapi 
 - frontend
 - action defined on precise datetime