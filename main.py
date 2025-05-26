
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from peewee import *
import os

DB = PostgresqlDatabase(
    os.environ.get("PGDATABASE", "railway"),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", "password"),
    host=os.environ.get("PGHOST", "localhost"),
    port=int(os.environ.get("PGPORT", 5432))
)

class BaseModelORM(Model):
    class Meta:
        database = DB

class Forecast(BaseModelORM):
    sku = CharField()
    time_key = IntegerField()
    pvp_is_competitorA = FloatField()
    pvp_is_competitorB = FloatField()
    pvp_is_competitorA_actual = FloatField(null=True)
    pvp_is_competitorB_actual = FloatField(null=True)

    class Meta:
        primary_key = CompositeKey('sku', 'time_key')

class ForecastRequest(BaseModel):
    sku: str
    time_key: int

class ActualPricesRequest(BaseModel):
    sku: str
    time_key: int
    pvp_is_competitorA_actual: float
    pvp_is_competitorB_actual: float

app = FastAPI()

@app.on_event("startup")
def startup():
    DB.connect()
    DB.create_tables([Forecast], safe=True)

@app.post("/forecast_prices/")
def get_forecast(req: ForecastRequest):
    record = Forecast.get_or_none((Forecast.sku == req.sku) & (Forecast.time_key == req.time_key))
    if not record:
        raise HTTPException(status_code=422, detail="SKU/time_key pair not found.")
    return {
        "sku": record.sku,
        "time_key": record.time_key,
        "pvp_is_competitorA": record.pvp_is_competitorA,
        "pvp_is_competitorB": record.pvp_is_competitorB
    }

@app.post("/actual_prices/")
def update_actual_prices(req: ActualPricesRequest):
    record = Forecast.get_or_none((Forecast.sku == req.sku) & (Forecast.time_key == req.time_key))
    if not record:
        raise HTTPException(status_code=422, detail="SKU/time_key pair not found.")
    record.pvp_is_competitorA_actual = req.pvp_is_competitorA_actual
    record.pvp_is_competitorB_actual = req.pvp_is_competitorB_actual
    record.save()
    return {
        "sku": record.sku,
        "time_key": record.time_key,
        "pvp_is_competitorA": record.pvp_is_competitorA,
        "pvp_is_competitorB": record.pvp_is_competitorB,
        "pvp_is_competitorA_actual": record.pvp_is_competitorA_actual,
        "pvp_is_competitorB_actual": record.pvp_is_competitorB_actual
    }
