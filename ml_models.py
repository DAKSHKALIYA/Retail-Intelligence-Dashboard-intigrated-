"""
Machine-learning layer for the Retail Intelligence Dashboard.

Run from the project root:   python ml_models.py

What it does
  1. K-Means customer clustering on log-scaled R/F/M (k chosen by silhouette score)
  2. Churn prediction  -> probability a customer will NOT buy in the next 90 days
  3. CLV prediction    -> expected revenue from the customer in the next 90 days
  4. Market-basket analysis -> products frequently bought together (support/confidence/lift)

Outputs
  ml_output/customer_scores.csv   one row per customer (used by Streamlit, FastAPI and Java)
  ml_output/cluster_summary.csv, kmeans_k_scores.csv, product_rules.csv,
  ml_output/feature_importance.csv, model_metrics.json
  models/churn_model.pkl, models/clv_model.pkl

Leakage-safe design: models are trained on a snapshot taken 90 days BEFORE the end of the
data (features use only older data, labels come from the last 90 days). The trained models
are then applied to features computed at the end of the data to score today's customers.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "Online Retail.xlsx"
RFM_FILE = ROOT / "rfm_customer_table.csv"
OUT_DIR = ROOT / "ml_output"
MODEL_DIR = ROOT / "models"

HORIZON_DAYS = 90
RANDOM_STATE = 42
FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "items_per_order",
    "n_products",
    "tenure_days",
    "avg_gap_days",
    "recency_gap_ratio",
    "orders_last_90d",
    "revenue_last_90d",
]


# --------------------------------------------------------------------------- data
def load_clean_transactions(path: Path = DATA_FILE) -> pd.DataFrame:
    """Same cleaning rules as data_utils.py, but without the Streamlit dependency."""
    df = pd.read_excel(path).drop_duplicates()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "InvoiceNo", "Description", "Quantity", "UnitPrice"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df = df[
        (~df["InvoiceNo"].str.startswith("C")) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    ].copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_features(tx: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Customer-level features using ONLY transactions before `as_of`."""
    d = tx[(tx["InvoiceDate"] < as_of) & tx["CustomerID"].notna()]
    inv = (
        d.groupby(["CustomerID", "InvoiceNo"])
        .agg(date=("InvoiceDate", "min"), rev=("Revenue", "sum"), items=("Quantity", "sum"))
        .reset_index()
    )
    g = inv.groupby("CustomerID")
    first, last = g["date"].min(), g["date"].max()

    f = pd.DataFrame(
        {
            "recency": (as_of - last).dt.days,
            "frequency": g["InvoiceNo"].nunique(),
            "monetary": g["rev"].sum(),
            "avg_order_value": g["rev"].mean(),
            "items_per_order": g["items"].mean(),
            "tenure_days": (as_of - first).dt.days,
        }
    )
    f["n_products"] = d.groupby("CustomerID")["StockCode"].nunique()
    gap = (last - first).dt.days / (f["frequency"] - 1).replace(0, np.nan)
    f["avg_gap_days"] = gap.fillna(f["tenure_days"])  # single purchase -> no repeat seen yet
    # how overdue is the customer compared with their own usual buying rhythm?
    f["recency_gap_ratio"] = f["recency"] / f["avg_gap_days"].clip(lower=1)
    recent = inv[inv["date"] >= as_of - pd.Timedelta(days=90)].groupby("CustomerID")
    f["orders_last_90d"] = recent["InvoiceNo"].nunique()
    f["revenue_last_90d"] = recent["rev"].sum()
    f[["orders_last_90d", "revenue_last_90d"]] = f[["orders_last_90d", "revenue_last_90d"]].fillna(0)
    f.index = f.index.astype(int)
    f.index.name = "CustomerID"
    return f[FEATURES]


def future_labels(tx: pd.DataFrame, start: pd.Timestamp, days: int) -> pd.DataFrame:
    w = tx[(tx["InvoiceDate"] >= start) & (tx["InvoiceDate"] < start + pd.Timedelta(days=days))]
    w = w[w["CustomerID"].notna()]
    return w.groupby(w["CustomerID"].astype(int))["Revenue"].sum().rename("future_revenue")


# ------------------------------------------------------------------- 1. clustering
def run_kmeans(rfm: pd.DataFrame):
    X = np.log1p(rfm[["Recency", "Frequency", "Monetary"]].clip(lower=0))
    Xs = StandardScaler().fit_transform(X)

    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xs)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Xs, km.labels_)})
    k_scores = pd.DataFrame(rows)
    best_k = int(k_scores[k_scores["k"].between(3, 6)].sort_values("silhouette").iloc[-1]["k"])

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(Xs)
    out = rfm.copy()
    out["Cluster"] = km.labels_

    # Give clusters human-readable names from their average behaviour.
    prof = out.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    z = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)
    prof["score"] = z["Frequency"] + z["Monetary"] - z["Recency"]
    order = prof.sort_values("score", ascending=False).index.tolist()
    names = {}
    for rank, c in enumerate(order):
        if rank == 0:
            names[c] = "High Value Active"
        elif rank == len(order) - 1:
            names[c] = "Dormant Low Value"
        else:
            names[c] = "Active Mid Value" if prof.loc[c, "Recency"] <= prof["Recency"].mean() else "Fading Mid Value"
    # make names unique if several middle clusters share one
    seen = {}
    for c in order:
        n = names[c]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} {seen[n]}"
    out["ClusterName"] = out["Cluster"].map(names)

    summary = (
        out.groupby(["Cluster", "ClusterName"])
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Monetary=("Monetary", "sum"),
        )
        .reset_index()
    )
    summary["Revenue_Share_%"] = summary["Total_Monetary"] / summary["Total_Monetary"].sum() * 100
    return out, summary, k_scores, best_k


# ---------------------------------------------------- 2 & 3. churn + CLV models
def train_churn_and_clv(tx: pd.DataFrame):
    end = tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    snap = end - pd.Timedelta(days=HORIZON_DAYS)

    X = build_features(tx, snap)
    fut = future_labels(tx, snap, HORIZON_DAYS)
    y_rev = fut.reindex(X.index).fillna(0.0)
    y_churn = (y_rev == 0).astype(int)

    Xtr, Xte, ytr, yte, rtr, rte = train_test_split(
        X, y_churn, y_rev, test_size=0.2, stratify=y_churn, random_state=RANDOM_STATE
    )

    # ---- churn classifier
    clf = GradientBoostingClassifier(
        n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=RANDOM_STATE
    ).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    churn_metrics = {
        "roc_auc": float(roc_auc_score(yte, p)),
        "baseline_recency_only_auc": float(roc_auc_score(yte, Xte["recency"])),
        "precision": float(precision_score(yte, pred)),
        "recall": float(recall_score(yte, pred)),
        "f1": float(f1_score(yte, pred)),
        "churn_rate_in_snapshot": float(y_churn.mean()),
        "train_customers": int(len(Xtr)),
        "test_customers": int(len(Xte)),
    }

    # ---- CLV regressor (log target because revenue is heavily skewed)
    reg = GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=RANDOM_STATE
    ).fit(Xtr, np.log1p(rtr))
    pr = np.expm1(reg.predict(Xte)).clip(min=0)
    clv_metrics = {
        "r2_log_scale": float(r2_score(np.log1p(rte), np.log1p(pr))),
        "mae_revenue": float(mean_absolute_error(rte, pr)),
        "mean_actual_revenue": float(rte.mean()),
    }

    importance = (
        pd.DataFrame({"feature": FEATURES, "churn_importance": clf.feature_importances_,
                      "clv_importance": reg.feature_importances_})
        .sort_values("churn_importance", ascending=False)
    )

    # ---- score customers "today" (features at end of data)
    X_now = build_features(tx, end)
    scores = pd.DataFrame(index=X_now.index)
    scores["ChurnProb"] = clf.predict_proba(X_now)[:, 1]
    scores["PredCLV90"] = np.expm1(reg.predict(X_now)).clip(min=0)

    meta = {"features": FEATURES, "horizon_days": HORIZON_DAYS, "trained_at": datetime.now().isoformat(timespec="seconds"),
            "data_end": str(end.date())}
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": clf, **meta}, MODEL_DIR / "churn_model.pkl")
    joblib.dump({"model": reg, **meta}, MODEL_DIR / "clv_model.pkl")
    return scores, churn_metrics, clv_metrics, importance, X_now


# ----------------------------------------------------- 4. market-basket analysis
def product_rules(tx: pd.DataFrame, top_n: int = 200, min_support: float = 0.01) -> pd.DataFrame:
    real = tx[tx["StockCode"].str.match(r"^\d{5}")]  # drop POST, DOT, M, BANK CHARGES ...
    name = real.groupby("StockCode")["Description"].agg(lambda s: s.mode().iloc[0])
    top = real.groupby("StockCode")["InvoiceNo"].nunique().nlargest(top_n).index
    b = real[real["StockCode"].isin(top)]
    basket = (b.pivot_table(index="InvoiceNo", columns="StockCode", values="Quantity", aggfunc="sum") > 0)
    M = basket.fillna(False).to_numpy(dtype=np.float32)
    n = M.shape[0]
    co = M.T @ M
    sup_item = np.diag(co) / n
    codes = basket.columns.to_numpy()

    rows = []
    ii, jj = np.where(np.triu(co, 1) / n >= min_support)
    for i, j in zip(ii, jj):
        s = co[i, j] / n
        for a, c in ((i, j), (j, i)):
            conf = s / sup_item[a]
            rows.append(
                {
                    "antecedent": name[codes[a]],
                    "consequent": name[codes[c]],
                    "support": s,
                    "confidence": conf,
                    "lift": conf / sup_item[c],
                }
            )
    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules
    return rules[rules["lift"] > 1].sort_values("lift", ascending=False).head(200).reset_index(drop=True)


# ------------------------------------------------------------------------ driver
def risk_band(p: float) -> str:
    return "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"


def run_all() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading and cleaning transactions ...")
    tx = load_clean_transactions()
    rfm = pd.read_csv(RFM_FILE)

    print("K-Means clustering ...")
    clustered, cluster_summary, k_scores, best_k = run_kmeans(rfm)

    print("Training churn + CLV models ...")
    scores, churn_m, clv_m, importance, X_now = train_churn_and_clv(tx)

    print("Market-basket analysis ...")
    rules = product_rules(tx)

    out = clustered.merge(scores, left_on="CustomerID", right_index=True, how="left")
    out["ChurnProb"] = out["ChurnProb"].fillna(out["ChurnProb"].median())
    out["PredCLV90"] = out["PredCLV90"].fillna(0.0)
    out["ChurnRisk"] = out["ChurnProb"].apply(risk_band)
    keep = ["CustomerID", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score",
            "Segment", "Cluster", "ClusterName", "ChurnProb", "ChurnRisk", "PredCLV90"]
    out[keep].round(4).to_csv(OUT_DIR / "customer_scores.csv", index=False)

    cluster_summary.round(2).to_csv(OUT_DIR / "cluster_summary.csv", index=False)
    k_scores.round(4).to_csv(OUT_DIR / "kmeans_k_scores.csv", index=False)
    importance.round(4).to_csv(OUT_DIR / "feature_importance.csv", index=False)
    rules.round(4).to_csv(OUT_DIR / "product_rules.csv", index=False)

    metrics = {"best_k": best_k, "churn": churn_m, "clv": clv_m,
               "high_risk_customers": int((out["ChurnRisk"] == "High").sum()),
               "total_customers": int(len(out))}
    (OUT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== Done ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run_all()
