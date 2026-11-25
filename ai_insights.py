"""
AI layer: natural-language answers and marketing copy built on top of the ML outputs.

Works in two modes:
  * LLM mode      - if ANTHROPIC_API_KEY is set (env var or Streamlit secrets), calls the Claude API
  * Offline mode  - otherwise falls back to rule-based answers, so the app never breaks
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "ml_output"
MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

SEGMENT_PLAYBOOK = {
    "Champions": "Reward loyalty: VIP perks, early access to new products, referral rewards.",
    "Loyal Customers": "Upsell bundles and keep engagement high with personalised offers.",
    "New Customers": "Drive the second purchase with a welcome series and a time-limited incentive.",
    "Big Spenders (New/Occasional)": "Personal follow-up and premium cross-sell suggestions.",
    "At Risk": "Urgent win-back campaign with a targeted offer based on past purchases.",
    "Cannot Lose Them": "High-touch outreach: personal call or email from an account manager.",
    "Promising": "Low-friction repeat-purchase reminders and recommendations.",
    "Hibernating": "Low-cost periodic reactivation; avoid expensive ad spend.",
    "Needs Attention": "Lifecycle nudges to move customers towards the loyal tier.",
}


def get_api_key() -> str | None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key
    try:  # Streamlit secrets (optional)
        import streamlit as st

        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def llm_available() -> bool:
    return bool(get_api_key())


def _call_llm(system: str, user: str, max_tokens: int = 700) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=get_api_key())
    msg = client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


# ----------------------------------------------------------------- context builder
def build_context(scores: pd.DataFrame) -> dict:
    seg = (
        scores.groupby("Segment")
        .agg(customers=("CustomerID", "count"), revenue=("Monetary", "sum"),
             avg_churn_prob=("ChurnProb", "mean"), predicted_clv_90d=("PredCLV90", "sum"))
        .round(2)
        .reset_index()
        .to_dict("records")
    )
    at_risk = (
        scores[scores["ChurnRisk"] == "High"]
        .assign(value_at_risk=lambda d: d["ChurnProb"] * d["Monetary"])
        .nlargest(10, "value_at_risk")[["CustomerID", "Segment", "Monetary", "ChurnProb"]]
        .round(2)
        .to_dict("records")
    )
    metrics = {}
    mp = OUT_DIR / "model_metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text())
    return {
        "total_customers": int(len(scores)),
        "total_revenue": round(float(scores["Monetary"].sum()), 2),
        "high_churn_risk_customers": int((scores["ChurnRisk"] == "High").sum()),
        "segments": seg,
        "top_10_high_value_at_risk": at_risk,
        "model_metrics": metrics,
    }


# --------------------------------------------------------------------- assistant
def answer_question(question: str, scores: pd.DataFrame) -> tuple[str, str]:
    """Returns (answer, mode) where mode is 'llm' or 'offline'."""
    ctx = build_context(scores)
    if llm_available():
        system = (
            "You are a retail analytics assistant. Answer ONLY from the JSON data provided. "
            "Be concise, use numbers, and say so if the data cannot answer the question. "
            "Churn probability = chance the customer makes no purchase in the next 90 days."
        )
        try:
            return _call_llm(system, f"DATA:\n{json.dumps(ctx, default=str)}\n\nQUESTION: {question}"), "llm"
        except Exception as e:  # network / key problems -> graceful fallback
            return _offline_answer(question, scores, ctx) + f"\n\n_(LLM unavailable: {type(e).__name__}; showing offline answer.)_", "offline"
    return _offline_answer(question, scores, ctx), "offline"


def _offline_answer(question: str, scores: pd.DataFrame, ctx: dict) -> str:
    q = question.lower()
    if any(w in q for w in ["risk", "churn", "leave", "lose"]):
        top = ctx["top_10_high_value_at_risk"][:5]
        lines = [f"- Customer {c['CustomerID']} ({c['Segment']}): spent {c['Monetary']:,.0f}, churn probability {c['ChurnProb']:.0%}" for c in top]
        return (f"{ctx['high_churn_risk_customers']:,} of {ctx['total_customers']:,} customers are high churn risk. "
                "Highest value at risk:\n" + "\n".join(lines))
    if any(w in q for w in ["segment", "revenue", "best", "top"]):
        seg = sorted(ctx["segments"], key=lambda s: s["revenue"], reverse=True)[:3]
        lines = [f"- {s['Segment']}: {s['customers']:,} customers, revenue {s['revenue']:,.0f}" for s in seg]
        return "Top segments by revenue:\n" + "\n".join(lines)
    if any(w in q for w in ["future", "clv", "value", "predict"]):
        total = scores["PredCLV90"].sum()
        return f"Models predict about {total:,.0f} in revenue from existing customers over the next 90 days."
    return ("I can answer questions about churn risk, segments, revenue and predicted value. "
            "Add an ANTHROPIC_API_KEY to enable free-form AI answers.")


# ------------------------------------------------------------- campaign generator
def campaign_for_segment(segment: str, stats: dict) -> tuple[str, str]:
    base = SEGMENT_PLAYBOOK.get(segment, "Maintain targeted lifecycle communication.")
    if llm_available():
        system = "You are a CRM marketing specialist for an online gift retailer. Be specific and practical."
        user = (
            f"Segment: {segment}\nStats: {json.dumps(stats, default=str)}\nStrategy hint: {base}\n"
            "Write: (1) a one-line goal, (2) an email subject line and 80-word email body, "
            "(3) two follow-up actions, (4) how to measure success."
        )
        try:
            return _call_llm(system, user, 600), "llm"
        except Exception:
            pass
    text = (
        f"**Goal:** {base}\n\n**Email subject:** We have something special for you\n\n"
        f"**Email body:** Hi there - as one of our {segment.lower()} we wanted to say thank you. "
        "Here is a personalised offer based on what you have enjoyed before.\n\n"
        "**Follow-ups:** A reminder email after 5 days; a small incentive if there is no response.\n\n"
        "**Measure:** Open rate, repeat-purchase rate within 30 days."
    )
    return text, "offline"
