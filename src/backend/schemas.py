import datetime as dt
from pydantic import BaseModel, Field
from backend.enums import Town, FlatType

class LinearRegressionModel_Input(BaseModel):
    model_config = {"extra": "forbid"} # forbid additional input

    date: dt.datetime
    floor_area_sqm: float = Field(gt=0, lt=2000) # largest floor_area_sqm is ~350 in training data
    town: Town
    flat_type: FlatType

class LinearRegressionModel_Output(BaseModel):
    model_config = {"extra": "forbid"} # forbid additional input

    resale_price: float
    # note: linear regression model can and is allowed to output negative values