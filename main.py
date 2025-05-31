import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import psycopg2
from typing import List, Optional

# ------------- Load Artifacts and Model -------------
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

# ------------- PostgreSQL Connection -------------
def get_pg_conn():
    return psycopg2.connect(
        dbname=os.getenv("PGDATABASE", "capsstone"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "1234"),
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", 5432)),
    )

# ------------- FastAPI Setup -------------
app = FastAPI(
    title="Competitor Price Forecasting API",
    description="Predict competitor prices using a LightGBM ensemble model"
)

# ------------- Pydantic Schemas -------------
class PredictionRequest(BaseModel):
    sku: str
    time_key: int
    pvp_was: float
    discount: float
    flag_promo: int
    leaflet: Optional[str] = "none"
    structure_level_1: int
    structure_level_2: int
    structure_level_3: int
    structure_level_4: int
    quantity: float
    year: int
    month: int
    day: int
    weekday: int
    quarter: int
    week_of_year: int
    is_campaign: int
    # Optionally: lag/rolling/comparison features if needed

class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest]

class PredictionResponse(BaseModel):
    pvp_is_competitorA: float
    pvp_is_competitorB: float

# ------------- Helper Functions -------------
def prepare_features(data: dict, feature_cols, leaflet_encoder):
    df = pd.DataFrame([data])
    # Leaflet encoding
    df["leaflet_encoded"] = df["leaflet"].fillna("none")
    df["leaflet_numeric"] = leaflet_encoder.transform(df["leaflet_encoded"])
    # Drop original leaflet, keep numeric
    df = df.drop(columns=["leaflet", "leaflet_encoded"], errors="ignore")
    # Fill missing columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    # Reorder
    df = df[feature_cols]
    df = df.fillna(0)
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

# ------------- API Endpoints -------------
@app.get("/")
def healthcheck():
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "msg": "Competitor Price Forecasting API running"}

@app.post("/predict/", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features = prepare_features(req.dict(), feature_cols, leaflet_encoder)
    preds = {}
    for comp in target_competitors:
        preds[comp] = float(forecaster.predict(features, comp)[0])
    # Store in DB
    upsert_forecast_to_db(req.sku, req.time_key, preds.get("competitorA", 0), preds.get("competitorB", 0))
    return PredictionResponse(
        pvp_is_competitorA=preds.get("competitorA", 0),
        pvp_is_competitorB=preds.get("competitorB", 0)
    )

@app.post("/predict/batch/", response_model=List[PredictionResponse])
def predict_batch(request: BatchPredictionRequest):
    results = []
    for item in request.items:
        features = prepare_features(item.dict(), feature_cols, leaflet_encoder)
        preds = {}
        for comp in target_competitors:
            preds[comp] = float(forecaster.predict(features, comp)[0])
        upsert_forecast_to_db(item.sku, item.time_key, preds.get("competitorA", 0), preds.get("competitorB", 0))
        results.append(PredictionResponse(
            pvp_is_competitorA=preds.get("competitorA", 0),
            pvp_is_competitorB=preds.get("competitorB", 0)
        ))
    return results

@app.get("/product_metadata/{sku}")
def get_product_metadata(sku: str):
    query = "SELECT * FROM product_metadata WHERE sku=%s"
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (sku,))
            row = cur.fetchone()
            if row:
                columns = [desc[0] for desc in cur.description]
                return dict(zip(columns, row))
            else:
                raise HTTPException(status_code=404, detail="SKU not found")
