import atexit
import logging
import random
from fastapi import FastAPI, Request, Depends, Query
from typing import Annotated
from contextlib import asynccontextmanager

from backend.model import LinearRegressionModel
from backend.config.logging_config import LOGGING_CONFIG
from backend.schemas import LinearRegressionModel_Input
from backend.database import database, get_session

def get_request_id():
    # there will be collisions but it should be very rare for simple debugging purposes.
    # if used for more complex purposes, use a better id generation method.
    randint = random.randint(1, 99999999)
    return f"R-{randint}"

# FastAPI Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI App Startup

    # Set up logging and log queue handler process
    logging.config.dictConfig(LOGGING_CONFIG)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
    app.state.logger = logging.getLogger(__name__)

    # Set up models
    app.state.lr = LinearRegressionModel()

    # Set up database
    database.setup_db()

    yield

    # FastAPI App Shutdown
    
    # Shut down database
    database.shutdown()

    # Shut down logging queue handler process
    if queue_handler is not None:
        queue_handler.listener.stop()

# FastAPI Dependencies
async def get_logger(request: Request):
    return request.app.state.logger
async def get_lr(request: Request):
    return request.app.state.lr
# get_session (imported) is also a dependency

# Set up API server
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}

# model inference
@app.get("/inference/linear-regression-model")
async def get_inference(
    input_features: Annotated[LinearRegressionModel_Input, Query(
        title="Linear Regression Model Input Features",
        description="All features (date, floor_area_sqm, town, flat_type) are required."
    )],
    logger: Annotated[Request, Depends(get_logger)],
    lr: Annotated[Request, Depends(get_lr)],
    session: Annotated[None, Depends(get_session)]
):
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
