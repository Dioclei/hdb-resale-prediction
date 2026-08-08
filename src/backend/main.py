from enum import Enum
from fastapi import FastAPI

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
    return {
        "item_id": item_id,
        "limit": limit,
        "location": location,
    }

# model inference
@app.get("/inference")
async def get_inference(town: TownName, model: FlatModelName):
    return {
        "town": town,
        "model": model
    }
