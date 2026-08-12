import atexit
import logging
from enum import Enum
from fastapi import FastAPI

from backend.config.logging_config import LOGGING_CONFIG

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)
setup_logging()
logger = logging.getLogger(__name__)

class TownName(str, Enum):
    AngMoKio = "Ang Mo Kio"
    Bishan = "Bishan"
    ToaPayoh = "Toa Payoh"

class FlatModelName(str, Enum):
    MultiGen = "Multi-Generation"
    Room5 = "5 Room"

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# example route: shows how path parameters and query parameters are parsed
@app.get("/items/{item_id}")
async def get_item(item_id: int, limit: int, location: str = "Singapore"):
    logger.info("Received item request", extra={
        "item_id": item_id,
        "limit": limit,
        "location": location
    })
    return {
        "item_id": item_id,
        "limit": limit,
        "location": location,
    }

# model inference
@app.get("/inference")
async def get_inference(town: TownName, model: FlatModelName):
    logger.info("Received inference request", extra={
        "town": town,
        "model": model,
    })
    return {
        "town": town,
        "model": model
    }
