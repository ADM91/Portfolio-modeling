#!/usr/bin/env python3
"""
Demonstration of the comprehensive logging system.
Run this file to test logging functionality.
"""

from utils.logging_config import setup_logging, get_logger, correlation_context, log_function_call
from database.access import DatabaseAccess
from database.init_db import initialize_database
from config import settings
import time

def test_basic_logging():
    """Test basic logging functionality."""
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")


def test_contextual_logging():
    """Test logging with context."""
    logger = get_logger(__name__)
    
    logger.info("Processing user data", extra={
        'context': {
            'user_id': 123,
            'operation': 'update_profile',
            'batch_size': 50
        }
    })


def test_performance_logging():
    """Test performance timing logging."""
    logger = get_logger(__name__)
    
    # Using context manager for timing
    with logger.perf.time_operation("database_query", {'table': 'users', 'filter': 'active'}):
        time.sleep(0.1)  # Simulate work
    
    # Using decorator for timing
    @logger.perf.time_function("user_processing")
    def process_user_data():
        time.sleep(0.05)  # Simulate work
        return "processed"
    
    result = process_user_data()
    logger.info(f"Function returned: {result}")


def test_correlation_context():
    """Test correlation ID context."""
    logger = get_logger(__name__)
    
    with correlation_context() as corr_id:
        logger.info("Starting operation with correlation ID", extra={
            'context': {'correlation_id': corr_id}
        })
        
        # Simulate multiple operations within the same request
        logger.info("Step 1: Validating input")
        logger.info("Step 2: Processing data")
        logger.info("Step 3: Saving results")


@log_function_call(get_logger(__name__), log_args=True, log_result=True)
def test_function_decoration(user_id: int, action: str):
    """Test automatic function call logging."""
    time.sleep(0.02)  # Simulate work
    return {"status": "success", "user_id": user_id, "action": action}


def test_error_logging():
    """Test error logging with context."""
    logger = get_logger(__name__)
    
    try:
        # Simulate an error
        raise ValueError("This is a simulated error for testing")
    except Exception as e:
        logger.error("Error occurred during processing", extra={
            'context': {
                'operation': 'test_error_logging',
                'error_type': type(e).__name__,
                'user_id': 456
            }
        }, exc_info=True)


def test_database_logging():
    """Test database operations with logging."""
    logger = get_logger(__name__)
    
    logger.info("Testing database operations with logging")
    
    try:
        # This will use the enhanced database logging
        db_access = DatabaseAccess()
        initialize_database(db_access)
        logger.info("Database operations completed successfully")
    except Exception as e:
        logger.error(f"Database operations failed: {str(e)}", exc_info=True)


def main():
    """Run all logging tests."""
    
    # Setup logging with pretty console format for demo
    setup_logging(
        log_level="DEBUG",
        log_format="pretty",
        log_file=None  # Console only for demo
    )
    
    logger = get_logger(__name__)
    logger.info("🧪 Starting logging system demonstration")
    
    print("\n" + "="*60)
    print("LOGGING SYSTEM DEMONSTRATION")
    print("="*60)
    
    print("\n1. Basic Logging Levels:")
    print("-" * 30)
    test_basic_logging()
    
    print("\n2. Contextual Logging:")
    print("-" * 30)
    test_contextual_logging()
    
    print("\n3. Performance Logging:")
    print("-" * 30)
    test_performance_logging()
    
    print("\n4. Correlation Context:")
    print("-" * 30)
    test_correlation_context()
    
    print("\n5. Function Decoration:")
    print("-" * 30)
    test_function_decoration(123, "update_profile")
    
    print("\n6. Error Logging:")
    print("-" * 30)
    test_error_logging()
    
    print("\n7. Database Operations Logging:")
    print("-" * 30)
    test_database_logging()
    
    print("\n" + "="*60)
    logger.info("✅ Logging system demonstration completed")
    print("="*60)
    
    print("\nKey Features Demonstrated:")
    print("- ✅ Colored console output with timestamps")
    print("- ✅ Contextual information in structured format")
    print("- ✅ Performance timing with automatic measurement")
    print("- ✅ Correlation IDs for request tracing")
    print("- ✅ Automatic function call logging")
    print("- ✅ Rich error logging with stack traces")
    print("- ✅ Database operation logging with detailed context")
    
    print(f"\nConfiguration used:")
    print(f"- Log Level: DEBUG")
    print(f"- Format: Pretty (development)")
    print(f"- Output: Console only")
    
    print("\n💡 For production, use:")
    print("- Log Level: INFO or WARNING")
    print("- Format: JSON (structured)")
    print("- Output: File with rotation")


if __name__ == "__main__":
    main()
