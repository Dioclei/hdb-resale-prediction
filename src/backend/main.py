import atexit
import logging
import datetime as dt
from enum import Enum
import random
from fastapi import FastAPI, Query, Path, Body
from pydantic import BaseModel
from typing import Literal, Annotated

from backend.model import LinearRegressionModel
from backend.config.logging_config import LOGGING_CONFIG

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

class TownName(str, Enum):
    ANG_MO_KIO = "ANG MO KIO"
    BEDOK = "BEDOK"
    BISHAN = "BISHAN"
    BUKIT_BATOK = "BUKIT BATOK"
    BUKIT_MERAH = "BUKIT MERAH"
    BUKIT_PANJANG = "BUKIT PANJANG"
    BUKIT_TIMAH = "BUKIT TIMAH"
    CENTRAL_AREA = "CENTRAL AREA"
    CHOA_CHU_KANG = "CHOA CHU KANG"
    CLEMENTI = "CLEMENTI"
    GEYLANG = "GEYLANG"
    HOUGANG = "HOUGANG"
    JURONG_EAST = "JURONG EAST"
    JURONG_WEST = "JURONG WEST"
    KALLANG_WHAMPOA = "KALLANG/WHAMPOA"
    MARINE_PARADE = "MARINE PARADE"
    PASIR_RIS = "PASIR RIS"
    PUNGGOL = "PUNGGOL"
    QUEENSTOWN = "QUEENSTOWN"
    SEMBAWANG = "SEMBAWANG"
    SENGKANG = "SENGKANG"
    SERANGOON = "SERANGOON"
    TAMPINES = "TAMPINES"
    TOA_PAYOH = "TOA PAYOH"
    WOODLANDS = "WOODLANDS"
    YISHUN = "YISHUN"

class FlatTypeName(str, Enum):
    ROOM_1 = "1 ROOM"
    ROOM_2 = "2 ROOM"
    ROOM_3 = "3 ROOM"
    ROOM_4 = "4 ROOM"
    ROOM_5 = "5 ROOM"
    EXECUTIVE = "EXECUTIVE"
    MULTI_GENERATION = "MULTI-GENERATION"

class InferenceInput_LinearRegressionModel(BaseModel):
    date: dt.datetime
    floor_area_sqm: float
    town: TownName
    flat_type: FlatTypeName

@app.get("/")
async def root():
    return {"message": "Hello World"}

# model inference
@app.get("/inference/linear-regression-model")
async def get_inference(
    input_features: Annotated[InferenceInput_LinearRegressionModel, Query(
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
    resale_price_pred = lr.predict(
        date=input_features.date,
        floor_area_sqm=input_features.floor_area_sqm,
        flat_type=input_features.flat_type, 
        town=input_features.town
    )
    logger.info(f"Inference output: {resale_price_pred}", extra={
        "request_id": req_id,
        "resale_price_pred": resale_price_pred,
    })

    return {
        "success": True,
        "input_features": input_features,
        "resale_price_pred": resale_price_pred,
    }
