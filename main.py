import os
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from peewee import *
from playhouse.db_url import connect
import holidays

# === Database Setup ===
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Environment variable DATABASE_URL is not set.")

# Connect to PostgreSQL using playhouse.db_url
DB = connect(DATABASE_URL)

# === Define Forecast Table Schema ===
class Forecast(Model):
    sku = CharField()
    time_key = IntegerField()
    pvp_is_competitora = FloatField()
    pvp_is_competitorb = FloatField()
    pvp_is_competitora_actual = FloatField(null=True)
    pvp_is_competitorb_actual = FloatField(null=True)

    class Meta:
        database = DB
        primary_key = CompositeKey("sku", "time_key")

# === Request Payload Schemas ===
class ForecastRequest(BaseModel):
    sku: str
    time_key: int  # Format: YYYYMMDD

class ActualPricesRequest(BaseModel):
    sku: str
    time_key: int
    pvp_is_competitora_actual: float
    pvp_is_competitorb_actual: float

# === Load Trained LightGBM Model ===
MODEL_PATH = "lightgbm_pipeline_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# === FastAPI App Initialization ===
app = FastAPI()

@app.on_event("startup")
def startup():
    DB.connect(reuse_if_open=True)
    DB.create_tables([Forecast], safe=True)

# === Endpoint: /forecast_prices/ ===
@app.post("/forecast_prices/")
def forecast_prices(req: ForecastRequest):
    # Validate date format
    try:
        dt = datetime.strptime(str(req.time_key), "%Y%m%d")
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid time_key format. Expected YYYYMMDD as integer.")

    # Extract calendar-based features
    month = dt.month
    dayofweek = dt.weekday()
    is_weekend = int(dayofweek >= 5)
    pt_holidays = holidays.CountryHoliday("PT", years=dt.year)
    is_holiday = int(dt in pt_holidays)

    # Construct input features
    base_features = {
        "sku": req.sku,
        "quantity": 1.0,
        "discount": 0.0,
        "flag_promo": "0",
        "leaflet": "None",
        "structure_level_1": "None",
        "structure_level_2": "None",
        "month": month,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "is_campaign": 0,
        "is_holiday": is_holiday,
    }

    # Predict for competitor A
    df_A = pd.DataFrame([{**base_features, "competitor": "competitora"}])
    pred_A = float(model.predict(df_A)[0])

    # Predict for competitor B
    df_B = pd.DataFrame([{**base_features, "competitor": "competitorb"}])
    pred_B = float(model.predict(df_B)[0])

    # Insert or update forecast in the database
    Forecast.insert({
    "sku": req.sku,
    "time_key": req.time_key,
    "pvp_is_competitora": pred_A,
    "pvp_is_competitorb": pred_B,
}).on_conflict(
    conflict_target=["sku", "time_key"],
    preserve=["sku", "time_key"],
    update={
        "pvp_is_competitora": pred_A,
        "pvp_is_competitorb": pred_B,
    }
).execute()

    return {
        "sku": req.sku,
        "time_key": req.time_key,
        "pvp_is_competitora": pred_A,
        "pvp_is_competitorb": pred_B,
    }

# === Endpoint: /actual_prices/ ===
@app.post("/actual_prices/")
def actual_prices(req: ActualPricesRequest):
    # Look up the forecast record
    record = Forecast.get_or_none((Forecast.sku == req.sku) & (Forecast.time_key == req.time_key))
    if not record:
        raise HTTPException(status_code=422, detail="Forecast not found for this SKU and date.")

    # Save the actual prices
    record.pvp_is_competitora_actual = req.pvp_is_competitorA_actual
    record.pvp_is_competitorb_actual = req.pvp_is_competitorB_actual
    record.save()

    return {
        "sku": record.sku,
        "time_key": record.time_key,
        "pvp_is_competitorA": record.pvp_is_competitora,
        "pvp_is_competitorB": record.pvp_is_competitorb,
        "pvp_is_competitorA_actual": record.pvp_is_competitora_actual,
        "pvp_is_competitorB_actual": record.pvp_is_competitorb_actual,
    }
