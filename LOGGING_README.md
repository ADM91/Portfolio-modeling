# Portfolio Tracker - Comprehensive Logging System

This document describes the comprehensive logging system implemented in the Portfolio Tracker application. The logging system provides structured, contextual, and performance-aware logging across all application layers.

## Features

### 🎨 **Dual Format Support**
- **Pretty Console Format**: Colored, human-readable logs for development
- **JSON Structured Format**: Machine-readable logs for production and log analysis

### 🔍 **Correlation ID Tracing**
- Automatic correlation ID generation for request tracing
- Track operations across multiple services and database calls
- Easy debugging of complex workflows

### ⏱️ **Performance Monitoring**
- Automatic timing of operations with context managers
- Function-level performance decoration
- Database query performance tracking

### 📊 **Contextual Information**
- Rich context data with every log entry
- Structured data for easy filtering and analysis
- Safe handling of sensitive information

### 🚀 **API Request Logging**
- Automatic logging of all HTTP requests and responses
- Request/response timing
- Error tracking with correlation IDs

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=pretty           # 'pretty' for dev, 'json' for production
LOG_FILE=logs/portfolio_tracker.log
LOG_MAX_SIZE=10485760      # 10MB
LOG_BACKUP_COUNT=5
LOG_SQL_QUERIES=false      # Enable SQL query logging
LOG_PERFORMANCE=true       # Enable performance timing logs
LOG_API_REQUESTS=true      # Enable API request/response logging

# Component-specific log levels
LOG_LEVEL_DATABASE=INFO
LOG_LEVEL_SERVICES=INFO
LOG_LEVEL_API=INFO
LOG_LEVEL_EXTERNAL=INFO    # For external API calls
```

### Python Configuration

```python
from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(
    log_level="INFO",
    log_format="pretty",  # or "json"
    log_file="logs/app.log",  # optional
    max_file_size=10 * 1024 * 1024,  # 10MB
    backup_count=5
)

# Get logger
logger = get_logger(__name__)
```

## Usage Examples

### Basic Logging

```python
from utils.logging_config import get_logger

logger = get_logger(__name__)

logger.debug("Detailed debugging information")
logger.info("General information about program execution")
logger.warning("Warning about potential issues")
logger.error("Error occurred but application continues")
logger.critical("Critical error, application may stop")
```

### Contextual Logging

```python
logger.info("Processing user data", extra={
    'context': {
        'user_id': 123,
        'operation': 'update_profile',
        'batch_size': 50,
        'source': 'api_endpoint'
    }
})
```

### Performance Timing

#### Using Context Manager

```python
# Time an operation with context
with logger.perf.time_operation("database_query", {'table': 'users', 'count': 1500}):
    # Your database operation here
    result = session.query(User).all()
```

#### Using Decorator

```python
# Time a function automatically
@logger.perf.time_function("user_validation")
def validate_user_data(user_data):
    # Function implementation
    return validated_data
```

### Correlation ID Context

```python
from utils.logging_config import correlation_context

with correlation_context() as corr_id:
    logger.info("Starting complex operation")
    # All logs within this context will have the same correlation ID
    process_step_1()
    process_step_2()
    process_step_3()
```

### Function Call Logging

```python
from utils.logging_config import log_function_call

@log_function_call(logger, log_args=True, log_result=False)
def process_transaction(transaction_id: str, amount: float):
    # Function implementation
    return result
```

### Error Logging

```python
try:
    risky_operation()
except Exception as e:
    logger.error("Operation failed", extra={
        'context': {
            'operation': 'risky_operation',
            'user_id': user.id,
            'error_type': type(e).__name__
        }
    }, exc_info=True)
```

## API Request Logging

The system automatically logs all API requests and responses when enabled:

```python
# In main.py, middleware is automatically added
if settings.log_api_requests:
    app.add_middleware(
        LoggingMiddleware,
        log_request_body=settings.debug,
        log_response_body=False
    )
```

### Request Log Example

```
→ POST /api/v1/portfolios [corr_id=a1b2c3d4]
```

### Response Log Example

```
← 200 POST /api/v1/portfolios [15.2ms] [corr_id=a1b2c3d4]
```

## Database Logging

Database operations are automatically logged with detailed context:

```python
# Example from database operations
logger.info("Completed insert_if_not_exists for Asset", extra={
    'context': {
        'model': 'Asset',
        'total_processed': 7,
        'inserted': 2,
        'skipped': 5
    }
})
```

## Log Formats

### Pretty Format (Development)

```
23:45:12.123 INFO     database.access:insert_if_not_exists:85 - Completed insert_if_not_exists for Asset [took 15.23ms] [model=Asset, total_processed=7, inserted=2, skipped=5] [corr_id=a1b2c3d4]
```

### JSON Format (Production)

```json
{
  "timestamp": "2024-01-15T23:45:12.123Z",
  "level": "INFO",
  "logger": "database.access",
  "message": "Completed insert_if_not_exists for Asset",
  "module": "access",
  "function": "insert_if_not_exists",
  "line": 85,
  "correlation_id": "a1b2c3d4",
  "duration_ms": 15.23,
  "context": {
    "model": "Asset",
    "total_processed": 7,
    "inserted": 2,
    "skipped": 5
  }
}
```

## Service-Specific Logging

### CLI Operations

```python
# CLI commands automatically use correlation context
with correlation_context() as corr_id:
    logger.info("Starting database initialization", extra={
        'context': {'operation': 'init_database', 'correlation_id': corr_id}
    })
```

### Database Access

```python
# Database operations include detailed logging
logger.info("Processing item 1/7 for Asset", extra={
    'context': {'item': asset_data, 'model': 'Asset'}
})
```

### Service Layer

Add logging to your services:

```python
from utils.logging_config import get_logger

class PortfolioService:
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def process_portfolio(self, portfolio_id: int):
        self.logger.info("Processing portfolio", extra={
            'context': {'portfolio_id': portfolio_id}
        })
        
        with self.logger.perf.time_operation("portfolio_processing", {'portfolio_id': portfolio_id}):
            # Your processing logic here
            pass
```

## Best Practices

### 1. Use Appropriate Log Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: General information about program execution
- **WARNING**: Something unexpected happened, but the application continues
- **ERROR**: A serious problem occurred
- **CRITICAL**: A very serious error occurred

### 2. Include Context

Always include relevant context information:

```python
# Good
logger.info("User login successful", extra={
    'context': {
        'user_id': user.id,
        'ip_address': request.client.host,
        'user_agent': request.headers.get('user-agent')
    }
})

# Bad
logger.info("User login successful")
```

### 3. Use Performance Timing

For operations that might be slow:

```python
with logger.perf.time_operation("complex_calculation", {'data_size': len(data)}):
    result = perform_complex_calculation(data)
```

### 4. Handle Sensitive Data

Never log sensitive information:

```python
# Good - exclude password
logger.info("User authentication attempt", extra={
    'context': {
        'username': username,
        'success': success
    }
})

# Bad - includes password
logger.info(f"Login attempt: {username}:{password}")
```

### 5. Use Correlation Context for Complex Operations

```python
with correlation_context() as corr_id:
    logger.info("Starting multi-step process")
    step1()
    step2()
    step3()
    logger.info("Multi-step process completed")
```

## Testing the Logging System

Run the comprehensive logging demo:

```bash
python logging_demo.py
```

This will demonstrate all logging features with example output.

## Monitoring and Analysis

### Log Analysis with JSON Format

When using JSON format, logs can be easily parsed and analyzed:

```bash
# Filter by correlation ID
grep '"correlation_id":"a1b2c3d4"' logs/portfolio_tracker.log

# Filter by performance (slow operations)
jq 'select(.duration_ms > 1000)' logs/portfolio_tracker.log

# Filter by errors
jq 'select(.level == "ERROR")' logs/portfolio_tracker.log

# Analyze by module
jq -r '.module' logs/portfolio_tracker.log | sort | uniq -c
```

### Integration with Log Management Systems

The JSON format is compatible with:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Fluentd**
- **Grafana Loki**
- **Datadog**
- **Splunk**

## Troubleshooting

### Common Issues

1. **Logs not appearing**: Check log level configuration
2. **Performance impact**: Reduce log level in production
3. **Large log files**: Ensure rotation is configured properly
4. **Missing correlation IDs**: Make sure to use `correlation_context()`

### Debug Configuration

For maximum debugging information:

```python
setup_logging(
    log_level="DEBUG",
    log_format="pretty",
    log_file="debug.log"
)
```

## File Structure

```
utils/
├── logging_config.py      # Core logging configuration
├── api_middleware.py      # FastAPI request logging middleware
└── ...

logs/                      # Log files directory
├── portfolio_tracker.log
├── portfolio_tracker.log.1
└── ...

logging_demo.py           # Comprehensive demo script
LOGGING_README.md        # This documentation
```

## Integration with Existing Code

The logging system is designed to work with existing code with minimal changes:

1. **Automatic CLI Logging**: CLI commands now have comprehensive logging
2. **Database Layer Logging**: All database operations are logged
3. **API Request Logging**: All HTTP requests are automatically logged
4. **Service Layer Ready**: Easy to add logging to service methods

---

This comprehensive logging system provides excellent visibility into your application's behavior, making debugging and monitoring much more effective.
