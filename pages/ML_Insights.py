from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import load_scores

OUT = Path(__file__).resolve().parent.parent / "ml_output"

st.set_page_config(page_title="ML Insights", page_icon=":robot_face:", layout="wide")
st.title("ML Insights")
st.caption("Page 3: Clustering, Churn Prediction, Customer Lifetime Value and Product Affinity")

try:
    scores, source = load_scores()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
st.sidebar.success(f"Data source: {source}")

tab1, tab2, tab3, tab4 = st.tabs(["Customer clusters", "Churn risk", "Predicted value (CLV)", "Products bought together"])

# ----------------------------------------------------------------- clusters
with tab1:
    st.subheader("K-Means customer clusters")
    ks = pd.read_csv(OUT / "kmeans_k_scores.csv")
    summary = pd.read_csv(OUT / "cluster_summary.csv")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(ks, x="k", y="silhouette", markers=True, template="plotly_white",
                      title="Silhouette score by number of clusters")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(ks, x="k", y="inertia", markers=True, template="plotly_white",
                      title="Elbow curve (inertia)")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(summary, use_container_width=True, hide_index=True)
    fig = px.scatter(scores, x="Frequency", y="Monetary", color="ClusterName", log_x=True, log_y=True,
                     hover_data=["CustomerID", "Recency", "Segment"], template="plotly_white",
                     title="Customers by cluster (log scale)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Rule-based RFM segment vs ML cluster**")
    cross = pd.crosstab(scores["Segment"], scores["ClusterName"])
    st.dataframe(cross, use_container_width=True)

# -------------------------------------------------------------------- churn
with tab2:
    st.subheader("Churn prediction (no purchase in the next 90 days)")
    metrics = json.loads((OUT / "model_metrics.json").read_text())
    ch = metrics["churn"]
    m = st.columns(4)
    m[0].metric("ROC-AUC", f"{ch['roc_auc']:.2f}", f"{ch['roc_auc'] - ch['baseline_recency_only_auc']:+.2f} vs recency-only")
    m[1].metric("Precision", f"{ch['precision']:.2f}")
    m[2].metric("Recall", f"{ch['recall']:.2f}")
    m[3].metric("High-risk customers", f"{metrics['high_risk_customers']:,}")

    risk_counts = scores["ChurnRisk"].value_counts().reindex(["High", "Medium", "Low"]).reset_index()
    risk_counts.columns = ["ChurnRisk", "Customers"]
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(risk_counts, x="ChurnRisk", y="Customers", color="ChurnRisk", template="plotly_white",
                     color_discrete_map={"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2ca02c"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fi = pd.read_csv(OUT / "feature_importance.csv").sort_values("churn_importance")
        fig = px.bar(fi, x="churn_importance", y="feature", orientation="h", template="plotly_white",
                     title="What drives churn predictions")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**High-value customers most worth saving** (churn probability x spend)")
    seg_filter = st.multiselect("Filter by segment", sorted(scores["Segment"].unique()))
    view = scores if not seg_filter else scores[scores["Segment"].isin(seg_filter)]
    view = view[view["ChurnRisk"] == "High"].assign(ValueAtRisk=lambda d: d["ChurnProb"] * d["Monetary"])
    st.dataframe(
        view.nlargest(50, "ValueAtRisk")[["CustomerID", "Segment", "Monetary", "Recency", "ChurnProb", "ValueAtRisk"]],
        use_container_width=True, hide_index=True,
    )

# ---------------------------------------------------------------------- CLV
with tab3:
    st.subheader("Predicted revenue per customer, next 90 days")
    clv = json.loads((OUT / "model_metrics.json").read_text())["clv"]
    m = st.columns(3)
    m[0].metric("Total predicted (90d)", f"{scores['PredCLV90'].sum():,.0f}")
    m[1].metric("Mean abs. error / customer", f"{clv['mae_revenue']:,.0f}")
    m[2].metric("R2 (log scale)", f"{clv['r2_log_scale']:.2f}")
    by_seg = scores.groupby("Segment", as_index=False)["PredCLV90"].sum().sort_values("PredCLV90")
    fig = px.bar(by_seg, x="PredCLV90", y="Segment", orientation="h", template="plotly_white",
                 title="Predicted 90-day revenue by segment")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(scores.nlargest(25, "PredCLV90")[["CustomerID", "Segment", "Monetary", "PredCLV90", "ChurnProb"]],
                 use_container_width=True, hide_index=True)

# ------------------------------------------------------------------- rules
with tab4:
    st.subheader("Frequently bought together (association rules)")
    rules = pd.read_csv(OUT / "product_rules.csv")
    min_lift = st.slider("Minimum lift", 1.0, float(max(rules["lift"].max(), 2.0)), 3.0)
    st.dataframe(rules[rules["lift"] >= min_lift].head(50), use_container_width=True, hide_index=True)
    st.caption("Lift > 1 means the products are bought together more often than chance. "
               "Use for bundles and 'customers also bought' recommendations.")
