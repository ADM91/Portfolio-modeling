import os
import sys
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from cli import db_update
from routers import metric_router, data_router
from config import settings
from utils.logging_config import setup_logging, get_logger
from utils.api_middleware import LoggingMiddleware

# Setup structured logging
setup_logging(
    log_level=settings.log_level,
    log_format=settings.log_format,
    log_file=settings.log_file if not settings.is_development else None,
    max_file_size=settings.log_max_size,
    backup_count=settings.log_backup_count
)

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Update data when server starts
    logger.info("🚀 Starting Portfolio Tracker API server", extra={
        'context': {
            'debug_mode': settings.debug,
            'host': settings.host,
            'port': settings.port,
            'log_level': settings.log_level,
            'database_url': settings.database_url
        }
    })
    
    try:
        with logger.perf.time_operation("startup_data_update"):
            db_update()  # Quick data refresh
        logger.info("✅ Startup data update completed successfully")
    except Exception as e:
        logger.error(f"⚠️ Startup data update failed: {str(e)}", exc_info=True)
    
    yield
    
    # Shutdown
    logger.info("🛑 Portfolio Tracker API server shutting down")


app = FastAPI(
    title="Portfolio Tracker API",
    description="Multi-currency portfolio tracking and analytics",
    debug=settings.debug,
    lifespan=lifespan
)

# Add logging middleware
if settings.log_api_requests:
    app.add_middleware(
        LoggingMiddleware,
        log_request_body=settings.debug,  # Only log request body in debug mode
        log_response_body=False  # Generally avoid logging response body due to size
    )

# Attach routers with prefix "api"
app.include_router(metric_router.router, prefix="/api/v1")
app.include_router(data_router.router, prefix="/api/v1")
