
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI()

class ForecastRequest(BaseModel):
    sku: str
    time_key: int

class ActualPriceRequest(BaseModel):
    sku: str
    time_key: int
    pvp_is_competitorA_actual: float
    pvp_is_competitorB_actual: float

@app.post("/forecast_prices/")
async def forecast_prices(req: ForecastRequest):
    return {
        "sku": req.sku,
        "time_key": req.time_key,
        "pvp_is_competitorA": 4.99,
        "pvp_is_competitorB": 4.79
    }

@app.post("/actual_prices/")
async def actual_prices(req: ActualPriceRequest):
    return {
        "sku": req.sku,
        "time_key": req.time_key,
        "pvp_is_competitorA": 4.99,
        "pvp_is_competitorB": 4.79,
        "pvp_is_competitorA_actual": req.pvp_is_competitorA_actual,
        "pvp_is_competitorB_actual": req.pvp_is_competitorB_actual
    }

@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, err: Exception):
    return JSONResponse(status_code=422, content={"error": str(err)})
