import os
import sys
import argparse
import uvicorn
from database.access_improved import DatabaseAccess
from database.init_db import initialize_database
from services.data_acquisition_service import DataAcquisitionService
from services.action_service import ActionService
from services.metric_service import MetricService
from services.portfolio_service import PortfolioService
from config import settings


def init_database():
    """Initialize the database with schema and reference data."""
    print("🗄️  Initializing database...")
    
    # Initialize the database
    db_access = DatabaseAccess()
    initialize_database(db_access)
    
    print("✅ Database initialized successfully")


def update_data():
    """Update asset price data and portfolio holdings."""
    print("📊 Updating market data...")
    
    db_access = DatabaseAccess()
    
    # YFinanceService get asset time series data
    yfinance_service = DataAcquisitionService(db_access)
    yfinance_service.update_db_with_asset_data()

    # ActionService get action data
    action_service = ActionService(db_access)
    # Use settings or environment variable for actions path
    actions_path = settings.path_actions
    if os.path.exists(actions_path):
        print(f"📋 Processing actions from {actions_path}...")
        actions = action_service.read_actions_from_excel(actions_path)
        action_service.insert_actions(actions)  # TODO: this reinserts existing actions
        action_service.process_actions()
        action_service.update_holdings_time_series_to_current_day()
    else:
        print(f"⚠️  Actions file not found at {actions_path}, skipping...")
        # Still update holdings time series for existing data
        action_service.update_holdings_time_series_to_current_day()

    print("✅ Data update completed")


def db_update():
    """Quick database update (used by API endpoints)."""
    db_access = DatabaseAccess()

    # Update asset price data
    yfinance_service = DataAcquisitionService(db_access)
    yfinance_service.update_db_with_asset_data()

    # Update portfolio holding time series
    action_service = ActionService(db_access)
    action_service.update_holdings_time_series_to_current_day()


def run_server():
    """Start the FastAPI server."""
    print(f"🚀 Starting server on {settings.host}:{settings.port}")
    print(f"🐛 Debug mode: {settings.debug}")
    print(f"📊 API docs will be available at: http://{settings.host}:{settings.port}/docs")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Portfolio Tracker CLI")
    parser.add_argument(
        "command",
        choices=["init", "update", "server", "full"],
        help="Command to run: init (database), update (data), server (start API), full (init + update + server)"
    )
    parser.add_argument(
        "--host",
        default=settings.host,
        help=f"Server host (default: {settings.host})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"Server port (default: {settings.port})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Override settings if provided
    if args.host != settings.host:
        settings.host = args.host
    if args.port != settings.port:
        settings.port = args.port
    if args.debug:
        settings.debug = True

    try:
        if args.command == "init":
            init_database()
        elif args.command == "update":
            update_data()
        elif args.command == "server":
            run_server()
        elif args.command == "full":
            init_database()
            update_data()
            run_server()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

