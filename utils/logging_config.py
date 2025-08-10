import os
import sys
import json
import time
import logging
import logging.handlers
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
import uuid


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter for production environments."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Create base log structure
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
            
        # Add performance timing if available
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
            
        # Add context data if available
        if hasattr(record, 'context'):
            log_data['context'] = record.context
            
        # Add error details if it's an exception
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
            
        return json.dumps(log_data, default=str)


class PrettyConsoleFormatter(logging.Formatter):
    """Pretty console formatter for development environments."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        if record.levelname in self.COLORS:
            colored_level = f"{self.COLORS[record.levelname]}{record.levelname:<8}{self.RESET}"
        else:
            colored_level = f"{record.levelname:<8}"
            
        # Build formatted message
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        formatted_msg = f"{timestamp} {colored_level} {record.module}:{record.funcName}:{record.lineno} - {record.getMessage()}"
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            formatted_msg += f" [corr_id={record.correlation_id}]"
            
        # Add duration if available
        if hasattr(record, 'duration_ms'):
            formatted_msg += f" [took {record.duration_ms:.2f}ms]"
            
        # Add context if available
        if hasattr(record, 'context'):
            context_str = ", ".join(f"{k}={v}" for k, v in record.context.items())
            formatted_msg += f" [{context_str}]"
            
        return formatted_msg


class CorrelationIdFilter(logging.Filter):
    """Filter to inject correlation ID into log records."""
    
    _local_correlation_id: Optional[str] = None
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        cls._local_correlation_id = correlation_id
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        return cls._local_correlation_id
    
    @classmethod
    def clear_correlation_id(cls):
        cls._local_correlation_id = None
    
    def filter(self, record: logging.LogRecord) -> bool:
        if self._local_correlation_id:
            record.correlation_id = self._local_correlation_id
        return True


class PerformanceLogger:
    """Utility class for performance logging."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @contextmanager
    def time_operation(self, operation_name: str, context: Optional[Dict[str, Any]] = None):
        """Context manager to time operations and log results."""
        start_time = time.perf_counter()
        
        self.logger.debug(f"Starting {operation_name}", extra={'context': context or {}})
        
        try:
            yield
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.logger.info(
                f"Completed {operation_name}",
                extra={
                    'duration_ms': duration_ms,
                    'context': context or {}
                }
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.logger.error(
                f"Failed {operation_name}: {str(e)}",
                extra={
                    'duration_ms': duration_ms,
                    'context': context or {}
                },
                exc_info=True
            )
            raise
    
    def time_function(self, operation_name: Optional[str] = None):
        """Decorator to time function execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                
                # Extract context from function arguments if possible
                context = {}
                if args and hasattr(args[0], '__class__'):
                    context['class'] = args[0].__class__.__name__
                
                with self.time_operation(name, context):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "pretty",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Setup application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('pretty' for console, 'json' for structured)
        log_file: Optional log file path
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if log_format.lower() == 'json':
        console_formatter = StructuredFormatter()
    else:
        console_formatter = PrettyConsoleFormatter()
    
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(correlation_filter)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        
        # Always use JSON format for file logging
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(correlation_filter)
        root_logger.addHandler(file_handler)
    
    # Log the setup completion
    logging.info(f"Logging configured: level={log_level}, format={log_format}, file={log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with performance timing capabilities."""
    logger = logging.getLogger(name)
    # Add performance logger as an attribute
    logger.perf = PerformanceLogger(logger)
    return logger


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """Context manager to set correlation ID for request tracing."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())[:8]
    
    CorrelationIdFilter.set_correlation_id(correlation_id)
    try:
        yield correlation_id
    finally:
        CorrelationIdFilter.clear_correlation_id()


def log_function_call(logger: logging.Logger, log_args: bool = False, log_result: bool = False):
    """
    Decorator to automatically log function calls.
    
    Args:
        logger: Logger instance to use
        log_args: Whether to log function arguments
        log_result: Whether to log function result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Prepare context
            context = {'function': func_name}
            
            if log_args:
                # Be careful not to log sensitive data
                safe_args = []
                for i, arg in enumerate(args):
                    if i == 0 and hasattr(arg, '__class__'):
                        safe_args.append(f"<{arg.__class__.__name__} instance>")
                    elif isinstance(arg, (str, int, float, bool)):
                        safe_args.append(arg)
                    else:
                        safe_args.append(f"<{type(arg).__name__}>")
                
                safe_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, (str, int, float, bool)) and 'password' not in k.lower():
                        safe_kwargs[k] = v
                    else:
                        safe_kwargs[k] = f"<{type(v).__name__}>"
                
                context.update({
                    'args': safe_args,
                    'kwargs': safe_kwargs
                })
            
            logger.debug(f"Calling {func_name}", extra={'context': context})
            
            try:
                result = func(*args, **kwargs)
                
                if log_result and result is not None:
                    if isinstance(result, (str, int, float, bool, list, dict)):
                        context['result'] = result
                    else:
                        context['result'] = f"<{type(result).__name__}>"
                
                logger.debug(f"Completed {func_name}", extra={'context': context})
                return result
                
            except Exception as e:
                logger.error(
                    f"Error in {func_name}: {str(e)}",
                    extra={'context': context},
                    exc_info=True
                )
                raise
                
        return wrapper
    return decorator
