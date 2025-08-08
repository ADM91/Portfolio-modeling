# Portfolio Dashboard

A modern, interactive Streamlit dashboard for portfolio analysis with multi-currency support.

## Features

### 🌍 Multi-Currency Base Asset Support
- Calculate all metrics in terms of any asset with price history
- Support for fiat currencies (USD, EUR, etc.)
- Support for cryptocurrencies (BTC, ETH, etc.)
- Support for commodities (Gold, Silver, etc.)
- Instant base asset switching with cached performance

### 📊 Three Core Metrics
1. **Holdings Value** - Portfolio value over time in selected base asset
2. **Value Invested** - Cumulative cash flows (money invested) over time
3. **Time-Weighted Returns** - Performance analysis with daily and cumulative returns

### 🎛️ Interactive Controls
- **Base Asset Selector** - Choose any asset as calculation base
- **Multi-Portfolio Selection** - Analyze multiple portfolios together
- **Multi-Asset Selection** - Filter specific assets or analyze all
- **Flexible Time Ranges** - Preset periods (30d, 90d, YTD, 1Y, Max) or custom dates
- **Real-time Updates** - Refresh data on demand

### 📈 Rich Visualizations
- Interactive Plotly charts with zoom, pan, and hover details
- Summary metric cards with key performance indicators
- Tabbed interface for different analysis views
- Performance statistics and breakdowns
- Responsive design for different screen sizes

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend
```bash
# Make sure your FastAPI server is running on http://localhost:8000
python main.py  # or however you start your FastAPI server
```

### 3. Launch the Dashboard
```bash
python run_dashboard.py
```

The dashboard will be available at: http://localhost:8501

## API Integration

The dashboard connects to your existing FastAPI endpoints:

- `/metrics/holdings-value` - Portfolio holdings value time series
- `/metrics/value-invested` - Investment cash flow time series  
- `/metrics/time-weighted-return` - Performance returns time series

All endpoints support:
- Multiple `portfolio_ids` and `asset_ids`
- Any `currency_id` (base asset for calculations)
- Flexible date ranges via `timeframe` or `start_date`/`end_date`

## Dashboard Structure

### Sidebar Controls
- **Base Asset Selection** - Prominent currency/asset selector
- **Portfolio Selection** - Multi-select for portfolio aggregation
- **Asset Selection** - Multi-select for asset filtering
- **Time Range** - Preset or custom date ranges
- **Refresh Button** - Clear cache and reload data

### Main Content
- **Summary Cards** - Key metrics at a glance
- **Holdings Value Tab** - Portfolio value over time
- **Investment Flow Tab** - Holdings vs invested comparison
- **Performance Tab** - Returns analysis with statistics

## Key Benefits

### For Multi-Portfolio Analysis
- Aggregate multiple portfolios as single entity
- Compare performance across portfolio combinations
- Analyze asset allocation across entire portfolio universe

### For Multi-Currency Analysis
- Instantly switch base currencies to see different perspectives
- Analyze portfolio performance in Bitcoin terms, gold terms, etc.
- No complex currency conversion - handled automatically by backend

### For Performance Analysis
- Time-weighted returns for accurate performance measurement
- Separate investment flows from market performance
- Rich performance statistics and risk metrics

## Technical Architecture

### Frontend (Streamlit)
- `frontend/dashboard.py` - Main dashboard application
- `frontend/api_client.py` - API communication layer
- `run_dashboard.py` - Startup script

### Backend Integration
- Uses existing FastAPI endpoints
- Leverages existing metric service with Redis caching
- No backend changes required - works with current infrastructure

### Caching Strategy
- Streamlit caching for API responses (5-minute TTL)
- Backend Redis caching for database queries
- Smart cache invalidation on data refresh

## Usage Examples

### Analyze All Portfolios in USD
1. Select "USD - US Dollar (Fiat)" as base asset
2. Select all portfolios
3. Select all assets
4. Choose "ytd" timeframe
5. View aggregated performance

### Compare Performance in Bitcoin Terms
1. Select "BTC - Bitcoin (Cryptocurrency)" as base asset
2. Select portfolios to compare
3. View how portfolios performed relative to Bitcoin

### Custom Date Range Analysis
1. Switch to "Custom" time range
2. Set specific start and end dates
3. Analyze performance over exact period

## Troubleshooting

### Dashboard Won't Load
- Ensure FastAPI server is running on http://localhost:8000
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify port 8501 is available

### No Data Showing
- Verify portfolios and assets are selected
- Check that selected portfolios contain the selected assets
- Ensure date range contains data
- Try refreshing data with the refresh button

### API Connection Errors
- Confirm FastAPI server is accessible at http://localhost:8000
- Check server logs for any errors
- Verify database is properly initialized and contains data

## Future Enhancements

- Real-time data updates via WebSocket
- Export functionality for charts and data
- Additional performance metrics (Sharpe ratio, drawdown analysis)
- Portfolio comparison tools
- Risk analysis dashboards
- Mobile-responsive improvements

## Development

The dashboard is built with:
- **Streamlit** - Web application framework
- **Plotly** - Interactive charting library
- **Pandas** - Data manipulation
- **Requests** - API communication

To modify or extend the dashboard, edit the files in the `frontend/` directory and restart the application.
