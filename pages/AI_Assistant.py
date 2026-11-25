from __future__ import annotations

import streamlit as st

from ai_insights import SEGMENT_PLAYBOOK, answer_question, campaign_for_segment, llm_available
from api_client import load_scores

st.set_page_config(page_title="AI Assistant", page_icon=":speech_balloon:", layout="wide")
st.title("AI Assistant")
st.caption("Page 4: Ask questions about your customers and generate marketing campaigns")

try:
    scores, source = load_scores()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if llm_available():
    st.sidebar.success("LLM mode: Claude API connected")
else:
    st.sidebar.warning("Offline mode. Set ANTHROPIC_API_KEY for full AI answers.")
st.sidebar.caption(f"Data source: {source}")

tab_chat, tab_campaign = st.tabs(["Ask your data", "Campaign generator"])

with tab_chat:
    st.write("Try: *Which high-value customers are about to leave?* or *Which segments earn the most?*")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(text)
    q = st.chat_input("Ask a question about customers, churn, segments or revenue")
    if q:
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans, mode = answer_question(q, scores)
            st.markdown(ans)
            st.caption(f"Answered in {mode} mode")
        st.session_state.chat.append(("assistant", ans))

with tab_campaign:
    seg = st.selectbox("Segment", sorted(SEGMENT_PLAYBOOK))
    sub = scores[scores["Segment"] == seg]
    stats = {
        "customers": int(len(sub)),
        "avg_spend": round(float(sub["Monetary"].mean()), 2) if len(sub) else 0,
        "avg_recency_days": round(float(sub["Recency"].mean()), 1) if len(sub) else 0,
        "avg_churn_probability": round(float(sub["ChurnProb"].mean()), 3) if len(sub) else 0,
    }
    st.json(stats)
    if st.button("Generate campaign"):
        with st.spinner("Writing campaign..."):
            text, mode = campaign_for_segment(seg, stats)
        st.markdown(text)
        st.caption(f"Generated in {mode} mode")
