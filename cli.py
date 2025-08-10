import os
import sys
import argparse
import uvicorn
from database.access import DatabaseAccess
from database.init_db import initialize_database
from services.data_acquisition_service import DataAcquisitionService
from services.transaction_processing_service import TransactionProcessingService
from config import settings
from utils.logging_config import setup_logging, get_logger, correlation_context

# Setup logging early for CLI operations
setup_logging(
    log_level=settings.log_level,
    log_format="pretty" if settings.is_development else settings.log_format,
    log_file=settings.log_file if not settings.is_development else None,
    max_file_size=settings.log_max_size,
    backup_count=settings.log_backup_count
)

logger = get_logger(__name__)


def init_database():
    """Initialize the database with schema and reference data."""
    with correlation_context() as corr_id:
        logger.info("🗄️ Starting database initialization", extra={
            'context': {'operation': 'init_database', 'correlation_id': corr_id}
        })
        
        try:
            with logger.perf.time_operation("database_initialization"):
                db_access = DatabaseAccess()
                initialize_database(db_access)
            
            logger.info("✅ Database initialized successfully")
            print("✅ Database initialized successfully")  # Keep console output for CLI users
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {str(e)}", exc_info=True)
            print(f"❌ Database initialization failed: {e}")
            raise


def update_data():
    """Update asset price data and portfolio holdings."""
    with correlation_context() as corr_id:
        logger.info("📊 Starting market data update", extra={
            'context': {'operation': 'update_data', 'correlation_id': corr_id}
        })
        
        try:
            db_access = DatabaseAccess()
            
            # Update asset price data
            logger.info("Updating asset price data from external sources")
            with logger.perf.time_operation("yfinance_data_update"):
                yfinance_service = DataAcquisitionService(db_access)
                yfinance_service.update_db_with_asset_data()
            
            # Process transaction data
            transaction_service = TransactionProcessingService(db_access)
            actions_path = settings.path_actions
            
            logger.info("Processing transaction data", extra={
                'context': {'actions_path': actions_path, 'file_exists': os.path.exists(actions_path)}
            })
            
            if os.path.exists(actions_path):
                print(f"📋 Processing actions from {actions_path}...")
                logger.info(f"Processing actions from Excel file", extra={
                    'context': {'file_path': actions_path}
                })
                
                with logger.perf.time_operation("excel_import"):
                    imported_count, import_errors = transaction_service.import_transactions_from_excel(actions_path)
                
                logger.info(f"Excel import completed", extra={
                    'context': {
                        'imported_count': imported_count,
                        'error_count': len(import_errors),
                        'errors': import_errors[:5] if import_errors else None
                    }
                })
                
                print(f"✅ Imported {imported_count} actions from Excel")
                if import_errors:
                    print(f"⚠️  Import warnings/errors: {len(import_errors)} issues found")
                    for error in import_errors[:5]:
                        print(f"   - {error}")
                    if len(import_errors) > 5:
                        print(f"   ... and {len(import_errors) - 5} more")
                
                with logger.perf.time_operation("transaction_processing"):
                    processed_count, processing_errors = transaction_service.process_unprocessed_transactions()
                
                logger.info(f"Transaction processing completed", extra={
                    'context': {
                        'processed_count': processed_count,
                        'error_count': len(processing_errors),
                        'errors': processing_errors[:5] if processing_errors else None
                    }
                })
                
                print(f"✅ Processed {processed_count} actions")
                if processing_errors:
                    print(f"⚠️  Processing warnings/errors: {len(processing_errors)} issues found")
                    for error in processing_errors[:5]:
                        print(f"   - {error}")
                    if len(processing_errors) > 5:
                        print(f"   ... and {len(processing_errors) - 5} more")
                
                with logger.perf.time_operation("holdings_time_series_update"):
                    transaction_service.update_holdings_time_series_to_current_day()
                
            else:
                logger.warning(f"Actions file not found", extra={
                    'context': {'expected_path': actions_path}
                })
                print(f"⚠️  Actions file not found at {actions_path}, skipping...")
                
                # Still update holdings time series for existing data
                with logger.perf.time_operation("holdings_time_series_update"):
                    transaction_service.update_holdings_time_series_to_current_day()
            
            logger.info("✅ Market data update completed successfully")
            print("✅ Data update completed")
            
        except Exception as e:
            logger.error(f"❌ Market data update failed: {str(e)}", exc_info=True)
            print(f"❌ Data update failed: {e}")
            raise


def db_update():
    """Quick database update (used by API endpoints)."""
    with correlation_context() as corr_id:
        logger.info("Starting quick database update", extra={
            'context': {'operation': 'db_update', 'correlation_id': corr_id}
        })
        
        try:
            db_access = DatabaseAccess()
            
            # Update asset price data
            with logger.perf.time_operation("quick_yfinance_update"):
                yfinance_service = DataAcquisitionService(db_access)
                yfinance_service.update_db_with_asset_data()
            
            # Update portfolio holding time series
            with logger.perf.time_operation("quick_holdings_update"):
                transaction_service = TransactionProcessingService(db_access)
                transaction_service.update_holdings_time_series_to_current_day()
            
            logger.info("Quick database update completed successfully")
            
        except Exception as e:
            logger.error(f"Quick database update failed: {str(e)}", exc_info=True)
            raise


def run_server():
    """Start the FastAPI server."""
    logger.info("🚀 Starting FastAPI server", extra={
        'context': {
            'host': settings.host,
            'port': settings.port,
            'debug': settings.debug,
            'log_level': settings.log_level
        }
    })
    
    print(f"🚀 Starting server on {settings.host}:{settings.port}")
    print(f"🐛 Debug mode: {settings.debug}")
    print(f"📊 API docs will be available at: http://{settings.host}:{settings.port}/docs")
    
    try:
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level=settings.log_level.lower()
        )
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}", exc_info=True)
        raise


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
