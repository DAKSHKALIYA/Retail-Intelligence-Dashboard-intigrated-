"""Loads customer scores from the Java Spring Boot API if it is running, otherwise from the CSV."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
SCORES_CSV = ROOT / "ml_output" / "customer_scores.csv"
JAVA_API_URL = os.getenv("JAVA_API_URL", "http://localhost:8080")


def java_api_up() -> bool:
    try:
        return requests.get(f"{JAVA_API_URL}/api/health", timeout=1.5).ok
    except requests.RequestException:
        return False


def load_scores(prefer_java: bool = True) -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, source) where source is 'Java API' or 'CSV file'."""
    if prefer_java and java_api_up():
        try:
            r = requests.get(f"{JAVA_API_URL}/api/customers", params={"limit": 100000}, timeout=20)
            r.raise_for_status()
            df = pd.DataFrame(r.json())
            df = df.rename(columns={
                "customerId": "CustomerID", "recency": "Recency", "frequency": "Frequency",
                "monetary": "Monetary", "segment": "Segment", "clusterName": "ClusterName",
                "churnProb": "ChurnProb", "churnRisk": "ChurnRisk", "predClv90": "PredCLV90",
            })
            return df, "Java API"
        except Exception:
            pass
    if not SCORES_CSV.exists():
        raise FileNotFoundError("ml_output/customer_scores.csv not found. Run:  python ml_models.py")
    return pd.read_csv(SCORES_CSV), "CSV file"
