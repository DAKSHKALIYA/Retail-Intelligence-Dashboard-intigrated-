"""
FastAPI ML service. The Java (Spring Boot) backend calls this.

Run from the project root:   uvicorn ml_service.main:app --port 8000
Docs:                        http://localhost:8000/docs
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="Retail ML Service", version="1.0")

if not (ROOT / "models" / "churn_model.pkl").exists():
    raise RuntimeError("Models not found. Run  python ml_models.py  from the project root first.")

SCORES = pd.read_csv(ROOT / "ml_output" / "customer_scores.csv").set_index("CustomerID")
CHURN = joblib.load(ROOT / "models" / "churn_model.pkl")
CLV = joblib.load(ROOT / "models" / "clv_model.pkl")


class CustomerFeatures(BaseModel):
    recency: float
    frequency: float
    monetary: float
    avg_order_value: float
    items_per_order: float
    n_products: float
    tenure_days: float
    avg_gap_days: float
    orders_last_90d: float = 0
    revenue_last_90d: float = 0


@app.get("/health")
def health():
    return {"status": "ok", "customers": int(len(SCORES)), "trained_at": CHURN["trained_at"]}


@app.get("/customers/{customer_id}")
def customer(customer_id: int):
    if customer_id not in SCORES.index:
        raise HTTPException(404, "Customer not found")
    row = SCORES.loc[customer_id].to_dict()
    row["CustomerID"] = customer_id
    return row


@app.post("/predict")
def predict(f: CustomerFeatures):
    """Score a brand-new feature vector (e.g. a what-if scenario)."""
    d = f.model_dump()
    d["recency_gap_ratio"] = d["recency"] / max(d["avg_gap_days"], 1)
    X = pd.DataFrame([d])[CHURN["features"]]
    return {
        "churn_probability": float(CHURN["model"].predict_proba(X)[0, 1]),
        "predicted_revenue_90d": float(max(0.0, np.expm1(CLV["model"].predict(X)[0]))),
    }
