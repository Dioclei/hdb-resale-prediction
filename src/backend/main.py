import atexit
import logging
import datetime as dt
import random
from fastapi import FastAPI, Query
from typing import Annotated

from backend.model import LinearRegressionModel
from backend.config.logging_config import LOGGING_CONFIG
from backend.schemas import LinearRegressionModel_Input

# Set up logging and log queue handler process
def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)
setup_logging()
logger = logging.getLogger(__name__)

# Set up models
lr = LinearRegressionModel()

# Set up API server
app = FastAPI()

def get_request_id():
    # there will be collisions but it should be very rare for simple debugging purposes.
    # if used for more complex purposes, use a better id generation method.
    randint = random.randint(1, 99999999)
    return f"R-{randint}"

@app.get("/")
async def root():
    return {"message": "Hello World"}

# model inference
@app.get("/inference/linear-regression-model")
async def get_inference(
    input_features: Annotated[LinearRegressionModel_Input, Query(
        title="Linear Regression Model Input Features",
        description="All features (date, floor_area_sqm, town, flat_type) are required."
    )]):
    req_id = get_request_id()
    logger.info("Inference request", extra={
        "request_id": req_id,
        "date": input_features.date,
        "floor_area_sqm": input_features.floor_area_sqm,
        "town": input_features.town,
        "flat_type": input_features.flat_type,
    })
    prediction = lr.predict(
        date=input_features.date,
        floor_area_sqm=input_features.floor_area_sqm,
        flat_type=input_features.flat_type, 
        town=input_features.town
    )
    logger.info(f"Inference output: {prediction.resale_price}", extra={
        "request_id": req_id,
        "resale_price_pred": prediction.resale_price,
    })

    # round output to 2 decimal places
    resale_price_pred = round(prediction.resale_price, 2)

    return {
        "success": True,
        "input_features": input_features,
        "resale_price_pred": resale_price_pred,
    }
