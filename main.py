
import os
import csv
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from playhouse.db_url import connect
from peewee import Model, CharField, IntegerField, FloatField

# Connect to Railway PostgreSQL or fallback to SQLite
DB = connect(os.environ.get("DATABASE_URL") or "sqlite:///predictions.db")

class Forecast(Model):
    sku = CharField()
    time_key = IntegerField()
    pvp_is_competitora = FloatField(null=True)
    pvp_is_competitorb = FloatField(null=True)

    class Meta:
        table_name = "forecast"
        database = DB
        primary_key = False

DB.connect()
DB.create_tables([Forecast])

# Auto-load CSV if DB is empty, then delete it
csv_path = Path("forecasts.csv")
if Forecast.select().count() == 0 and csv_path.exists():
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        with DB.atomic():
            for row in reader:
                Forecast.create(
                    sku=row["sku"],
                    time_key=int(row["time_key"]),
                    pvp_is_competitora=float(row["pvp_is_competitora"]) if row["pvp_is_competitora"] else None,
                    pvp_is_competitorb=float(row["pvp_is_competitorb"]) if row["pvp_is_competitorb"] else None
                )
    os.remove(csv_path)

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
def forecast_prices(req: ForecastRequest):
    try:
        record = Forecast.get((Forecast.sku == req.sku) & (Forecast.time_key == req.time_key))
        return {
            "sku": req.sku,
            "time_key": req.time_key,
            "pvp_is_competitorA": record.pvp_is_competitora,
            "pvp_is_competitorB": record.pvp_is_competitorb
        }
    except Forecast.DoesNotExist:
        raise HTTPException(status_code=422, detail="sku/time_key pair not found")

@app.post("/actual_prices/")
def actual_prices(req: ActualPriceRequest):
    try:
        record = Forecast.get((Forecast.sku == req.sku) & (Forecast.time_key == req.time_key))
        return {
            "sku": req.sku,
            "time_key": req.time_key,
            "pvp_is_competitorA": record.pvp_is_competitora,
            "pvp_is_competitorB": record.pvp_is_competitorb,
            "pvp_is_competitorA_actual": req.pvp_is_competitorA_actual,
            "pvp_is_competitorB_actual": req.pvp_is_competitorB_actual
        }
    except Forecast.DoesNotExist:
        raise HTTPException(status_code=422, detail="sku/time_key pair not found")
