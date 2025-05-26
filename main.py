
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from peewee import *
from playhouse.db_url import connect

# Load environment-based database connection
DB = connect(os.environ.get("DATABASE_URL"))

# Load pre-trained model pipeline
try:
    model = joblib.load("lightgbm_pipeline_model.pkl")
except Exception as e:
    raise RuntimeError("Model file could not be loaded: {}".format(str(e)))

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
    quantity: float = 1.0
    discount: float = 0.0
    flag_promo: str = "0"
    leaflet: str = "None"
    structure_level_1: str = "None"
    structure_level_2: str = "None"
    month: int = 1
    dayofweek: int = 0
    is_weekend: int = 0
    is_campaign: int = 0

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
def forecast_prices(req: ForecastRequest):
    features = pd.DataFrame([{
        "sku": req.sku,
        "competitor": "competitorA",
        "quantity": req.quantity,
        "discount": req.discount,
        "flag_promo": req.flag_promo,
        "leaflet": req.leaflet,
        "structure_level_1": req.structure_level_1,
        "structure_level_2": req.structure_level_2,
        "month": req.month,
        "dayofweek": req.dayofweek,
        "is_weekend": req.is_weekend,
        "is_campaign": req.is_campaign
    }])

    pred_A = model.predict(features)[0]

    features["competitor"] = "competitorB"
    pred_B = model.predict(features)[0]

    Forecast.insert({
    Forecast.sku: req.sku,
    Forecast.time_key: req.time_key,
    Forecast.pvp_is_competitorA: pred_A,
    Forecast.pvp_is_competitorB: pred_B,
}).on_conflict(
    on_conflict=peewee.ConflictWhere(
        (Forecast.sku == req.sku) & (Forecast.time_key == req.time_key)
    ),
    update={
        Forecast.pvp_is_competitorA: pred_A,
        Forecast.pvp_is_competitorB: pred_B,
    }
).execute()

    return {
        "sku": req.sku,
        "time_key": req.time_key,
        "pvp_is_competitorA": pred_A,
        "pvp_is_competitorB": pred_B
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
        "pvp_is_competitorB_actual": record.pvp_is_competitorB_actual,
    }
