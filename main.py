import os
import sys
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from cli import db_update
from routers import metric_router, data_router
from config import settings


# Configure logging using settings
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, settings.log_level),
    format="%(levelname)s: %(module)s: %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Update data when server starts
    print("🚀 Starting up...")
    try:
        db_update()  # Quick data refresh
    except Exception as e:
        print(f"⚠️ Startup data update failed: {e}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")


app = FastAPI(
    title="Portfolio Tracker API",
    description="Multi-currency portfolio tracking and analytics",
    debug=settings.debug,
    lifespan=lifespan
)

# Attach routers with prefix "api"
app.include_router(metric_router.router, prefix="/api/v1")
app.include_router(data_router.router, prefix="/api/v1")
