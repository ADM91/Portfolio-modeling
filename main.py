
import os
import sys
import logging
from fastapi import FastAPI

from routers import metric_router


# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(levelname)s: %(module)s: %(message)s",
)

app = FastAPI()

# Attach routers with prefix "api"
app.include_router(metric_router.router)



