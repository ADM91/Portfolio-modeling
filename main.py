import os
import sys
import logging
from fastapi import FastAPI

from routers import metric_router
from config import settings

# Configure logging using settings
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, settings.log_level),
    format="%(levelname)s: %(module)s: %(message)s",
)

app = FastAPI(
    title="Portfolio Tracker API",
    description="Multi-currency portfolio tracking and analytics",
    debug=settings.debug
)

# Attach routers with prefix "api"
app.include_router(metric_router.router)
