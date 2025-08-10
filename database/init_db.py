
from database.access import with_session, DatabaseAccess
from database.entities import TransactionType, Asset, Portfolio
from config import transaction_types, assets, portfolios, settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


@with_session
def initialize_database(session, db: DatabaseAccess):
    """Initialize database with schema and reference data."""
    logger.info("Starting database initialization")
    
    try:
        # Initialize database schema
        with logger.perf.time_operation("create_database_schema"):
            db.init_db()
        logger.info("✅ Database schema created successfully")
        
        # Insert transaction types
        logger.info("Inserting transaction types", extra={
            'context': {'count': len(transaction_types), 'data': transaction_types}
        })
        with logger.perf.time_operation("insert_transaction_types"):
            db.insert_if_not_exists(session, TransactionType, transaction_types)
        logger.info(f"✅ Transaction types processed: {len(transaction_types)} records")
        
        # Insert assets
        logger.info("Inserting assets", extra={
            'context': {'count': len(assets)}
        })
        # Log each asset being processed
        for i, asset in enumerate(assets):
            logger.debug(f"Processing asset {i+1}/{len(assets)}", extra={
                'context': {'asset': asset}
            })
        
        with logger.perf.time_operation("insert_assets"):
            db.insert_if_not_exists(session, Asset, assets)
        logger.info(f"✅ Assets processed: {len(assets)} records")
        
        # Insert portfolios
        logger.info("Inserting portfolios", extra={
            'context': {'count': len(portfolios), 'data': portfolios}
        })
        with logger.perf.time_operation("insert_portfolios"):
            db.insert_if_not_exists(session, Portfolio, portfolios)
        logger.info(f"✅ Portfolios processed: {len(portfolios)} records")
        
        logger.info("🎉 Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}", extra={
            'context': {
                'transaction_types_count': len(transaction_types),
                'assets_count': len(assets),
                'portfolios_count': len(portfolios)
            }
        }, exc_info=True)
        raise


if __name__ == "__main__":
    # Setup logging for standalone execution
    from utils.logging_config import setup_logging
    setup_logging(log_level="DEBUG", log_format="pretty")
    
    logger.info("Starting standalone database initialization")
    db = DatabaseAccess()
    initialize_database(db)
    logger.info("✅ Standalone database initialization completed")
