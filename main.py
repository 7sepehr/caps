import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import joblib
import psycopg2
import pandas as pd
from typing import Optional

# ------------------- Load Model Artifacts -------------------
MODEL_PATH = "saved_models/competitor_price_forecaster.pkl"
LEAFLET_ENCODER_PATH = "saved_models/leaflet_encoder.pkl"
FEATURE_COLUMNS_PATH = "saved_models/feature_columns.pkl"
TARGET_COMPETITORS_PATH = "saved_models/target_competitors.pkl"

def load_artifacts():
    try:
        forecaster = joblib.load(MODEL_PATH)
        leaflet_encoder = joblib.load(LEAFLET_ENCODER_PATH)
        feature_cols = joblib.load(FEATURE_COLUMNS_PATH)
        target_competitors = joblib.load(TARGET_COMPETITORS_PATH)
        return forecaster, leaflet_encoder, feature_cols, target_competitors
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None, None, None, None

forecaster, leaflet_encoder, feature_cols, target_competitors = load_artifacts()

# ------------------- DB Connection -------------------
def get_pg_conn():
    return psycopg2.connect(
        dbname=os.getenv("PGDATABASE", "capsstone"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "1234"),
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", 5432)),
    )

# ------------------- Helper Functions -------------------
def prepare_features(sku, time_key):
    """
    Given sku and time_key, look up or build a default feature row for prediction.
    This is a placeholder. In a real scenario, you would look up or compute all required features for prediction.
    """
    # Example: Try to look up product metadata from DB (you can expand this logic)
    metadata = None
    try:
        with get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT structure_level_1, structure_level_2, structure_level_3, structure_level_4,
                           last_discount, last_flag_promo, last_leaflet
                    FROM product_metadata
                    WHERE sku=%s
                    LIMIT 1
                """, (sku,))
                row = cur.fetchone()
                if row:
                    metadata = {
                        "structure_level_1": row[0],
                        "structure_level_2": row[1],
                        "structure_level_3": row[2],
                        "structure_level_4": row[3],
                        "discount": row[4],
                        "flag_promo": row[5],
                        "leaflet": row[6] or "none"
                    }
    except Exception as e:
        metadata = None

    # Build the full feature dict, with defaults if not found
    # You should update this to reflect your real feature engineering logic.
    # For demo, we fill with zeros/defaults.
    from datetime import datetime

    # Parse time_key to date
    try:
        date_obj = datetime.strptime(str(time_key), "%Y%m%d")
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        weekday = date_obj.weekday()
        quarter = (date_obj.month - 1)//3 + 1
        week_of_year = int(date_obj.strftime("%V"))  # ISO week
    except Exception:
        year = month = day = weekday = quarter = week_of_year = 0

    features = {
        "sku": sku,
        "pvp_was": 0.0,
        "discount": metadata["discount"] if metadata else 0.0,
        "flag_promo": metadata["flag_promo"] if metadata else 0,
        "leaflet": metadata["leaflet"] if metadata else "none",
        "structure_level_1": metadata["structure_level_1"] if metadata else 0,
        "structure_level_2": metadata["structure_level_2"] if metadata else 0,
        "structure_level_3": metadata["structure_level_3"] if metadata else 0,
        "structure_level_4": metadata["structure_level_4"] if metadata else 0,
        "quantity": 0.0,
        "year": year,
        "month": month,
        "day": day,
        "weekday": weekday,
        "quarter": quarter,
        "week_of_year": week_of_year,
        "is_campaign": 0,
        "time_key": time_key,
    }

    # Add any missing columns to match feature_cols
    for col in feature_cols:
        if col not in features:
            features[col] = 0

    # Leaflet encoding
    df = pd.DataFrame([features])
    df["leaflet_encoded"] = df["leaflet"].fillna("none")
    df["leaflet_numeric"] = leaflet_encoder.transform(df["leaflet_encoded"])
    df = df.drop(columns=["leaflet", "leaflet_encoded"], errors="ignore")
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols].fillna(0)
    return df

def upsert_forecast_to_db(sku, time_key, pvp_A, pvp_B):
    query = """
        INSERT INTO forecasts (sku, time_key, "pvp_is_competitorA", "pvp_is_competitorB")
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (sku, time_key) DO UPDATE
        SET "pvp_is_competitorA" = EXCLUDED."pvp_is_competitorA",
            "pvp_is_competitorB" = EXCLUDED."pvp_is_competitorB"
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (sku, time_key, pvp_A, pvp_B))
        conn.commit()

def upsert_actual_prices_to_db(sku, time_key, pvp_A_actual, pvp_B_actual):
    query = """
        UPDATE forecasts
        SET "pvp_is_competitorA_actual" = %s,
            "pvp_is_competitorB_actual" = %s
        WHERE sku=%s AND time_key=%s
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (pvp_A_actual, pvp_B_actual, sku, time_key))
            if cur.rowcount == 0:
                # If not exist, insert new row with actuals
                cur.execute("""
                    INSERT INTO forecasts (sku, time_key, "pvp_is_competitorA_actual", "pvp_is_competitorB_actual")
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (sku, time_key) DO NOTHING
                """, (sku, time_key, pvp_A_actual, pvp_B_actual))
        conn.commit()

# ------------------- Schemas -------------------
class ForecastReq(BaseModel):
    sku: str = Field(..., description="Product SKU")
    time_key: int = Field(..., description="Date as YYYYMMDD integer")

class ForecastResp(BaseModel):
    sku: str
    time_key: int
    pvp_is_competitorA: float
    pvp_is_competitorB: float

class ActualPriceReq(BaseModel):
    sku: str = Field(..., description="Product SKU")
    time_key: int = Field(..., description="Date as YYYYMMDD integer")
    pvp_is_competitorA_actual: float = Field(..., description="Actual price competitor A")
    pvp_is_competitorB_actual: float = Field(..., description="Actual price competitor B")

# ------------------- FastAPI App -------------------
app = FastAPI(
    title="Competitor Price Forecasting API",
    description="API to forecast and store competitor prices",
)

@app.post("/forecast_prices/", response_model=ForecastResp)
def forecast_prices(req: ForecastReq):
    # Validate input (Pydantic does this, but double check)
    if not req.sku or not isinstance(req.time_key, int):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid input format.")

    if forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = prepare_features(req.sku, req.time_key)
    preds = {}
    try:
        for comp in target_competitors:
            preds[comp] = float(forecaster.predict(X, comp)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed.")

    upsert_forecast_to_db(req.sku, req.time_key, preds.get("competitorA", 0), preds.get("competitorB", 0))
    return ForecastResp(
        sku=req.sku,
        time_key=req.time_key,
        pvp_is_competitorA=preds.get("competitorA", 0),
        pvp_is_competitorB=preds.get("competitorB", 0),
    )

@app.post("/actual_prices/")
def actual_prices(req: ActualPriceReq):
    if not req.sku or not isinstance(req.time_key, int):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid input format.")
    try:
        upsert_actual_prices_to_db(req.sku, req.time_key, req.pvp_is_competitorA_actual, req.pvp_is_competitorB_actual)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not save actual prices.")
    return {"status": "success"}
