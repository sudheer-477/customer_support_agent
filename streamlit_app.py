"""
streamlit_app.py — Care Intelligence support assistant (Streamlit UI)
------------------------------------------------------------------------
Real chat interface wired to the actual CrewAI + Groq flow in
care_intelligence_flow.py — this is not a mock. Every message runs the
full intent-detection + RAG + guardrail pipeline.

Setup:
    pip install -r requirements.txt
    pip install streamlit --break-system-packages
    export GROQ_API_KEY="gsk_..."

Run:
    streamlit run streamlit_app.py
"""

import time
import csv
import os
from datetime import datetime

import streamlit as st

from care_intelligence_flow import CareIntelligenceSupportFlow
from agent_core import INTENT_FAQ, INTENT_COMPLAINT, INTENT_ESCALATE, INTENT_OFF_TOPIC

FEEDBACK_LOG_PATH = os.path.join(os.path.dirname(__file__), "feedback_log.csv")

INTENT_STYLE = {
    INTENT_FAQ:        {"label": "FAQ",        "color": "#0C447C", "bg": "#E6F1FB"},
    INTENT_COMPLAINT:  {"label": "Complaint",  "color": "#854F0B", "bg": "#FAEEDA"},
    INTENT_ESCALATE:   {"label": "Escalated",  "color": "#791F1F", "bg": "#FCEBEB"},
    INTENT_OFF_TOPIC:  {"label": "Off-topic",  "color": "#444441", "bg": "#F1EFE8"},
}


# ---------------------------------------------------------------------------
# Page setup + branding
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Care Intelligence Support",
    page_icon=":hospital:",
    layout="centered",
)

st.markdown("""
<style>
.ci-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 0 20px 0;
    border-bottom: 1px solid #e5e5e5;
    margin-bottom: 20px;
}
.ci-logo {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: #0C447C;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 18px;
    flex-shrink: 0;
}
.ci-title { font-weight: 600; font-size: 20px; margin: 0; }
.ci-subtitle { font-size: 13px; color: #666; margin: 0; }
.intent-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 12px;
    margin-right: 6px;
}
.meta-line { font-size: 11px; color: #888; margin-top: 4px; }
.escalation-box {
    background: #FCEBEB;
    border: 1px solid #f0a3a3;
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 6px;
    font-size: 13px;
    color: #791F1F;
}
</style>
<div class="ci-header">
    <div class="ci-logo">CI</div>
    <div>
        <p class="ci-title">Care Intelligence</p>
        <p class="ci-subtitle">AI support assistant for healthcare centers</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — status + guardrail transparency
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Status")
    key_set = bool(os.environ.get("GROQ_API_KEY"))
    st.markdown(f"**Groq API key:** {'connected' if key_set else 'not set'}")
    if not key_set:
        st.warning("Set GROQ_API_KEY in your environment before chatting.")

    st.markdown("### About this assistant")
    st.caption(
        "Routes every message through intent detection, then one of four "
        "flows: FAQ retrieval, complaint handling, escalation, or a "
        "polite off-topic refusal. See GUARDRAILS.md for full details."
    )

    st.markdown("### Session stats")
    total = len(st.session_state.get("messages", []))
    st.metric("Messages this session", total // 2 if total else 0)

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: role, content, meta

if "feedback" not in st.session_state:
    st.session_state.feedback = {}  # message_index -> "up" | "down"


def log_feedback(message_index: int, query: str, response: str, intent: str, value: str):
    st.session_state.feedback[message_index] = value
    file_exists = os.path.exists(FEEDBACK_LOG_PATH)
    with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "query", "response", "intent", "feedback"])
        writer.writerow([datetime.utcnow().isoformat(), query, response, intent, value])


# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            meta = msg["meta"]
            style = INTENT_STYLE[meta["intent"]]
            st.markdown(
                f'<span class="intent-badge" style="color:{style["color"]}; '
                f'background:{style["bg"]};">{style["label"]}</span>'
                f'<span class="meta-line">{meta["response_time_ms"]} ms · '
                f'confidence {meta["confidence"]:.2f}</span>',
                unsafe_allow_html=True,
            )

            if meta["intent"] == INTENT_ESCALATE:
                st.markdown(
                    f'<div class="escalation-box">'
                    f'<strong>Connecting to a human agent</strong><br>{msg["content"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.write(msg["content"])

            col1, col2, _ = st.columns([1, 1, 8])
            existing = st.session_state.feedback.get(i)
            with col1:
                if st.button("Helpful", key=f"up_{i}",
                              type="primary" if existing == "up" else "secondary"):
                    log_feedback(i, meta["query"], msg["content"], meta["intent"], "up")
                    st.rerun()
            with col2:
                if st.button("Not helpful", key=f"down_{i}",
                              type="primary" if existing == "down" else "secondary"):
                    log_feedback(i, meta["query"], msg["content"], meta["intent"], "down")
                    st.rerun()
            if existing:
                st.caption(f"You marked this as {'helpful' if existing == 'up' else 'not helpful'}.")
        else:
            st.write(msg["content"])


# ---------------------------------------------------------------------------
# Chat input -> run the real flow
# ---------------------------------------------------------------------------

query = st.chat_input("Ask about our products, pricing, security, integrations...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            try:
                flow = CareIntelligenceSupportFlow(query)
                flow.kickoff()
                state = flow.state
                response_text = state.response
                intent = state.intent
                confidence = state.confidence
            except Exception as e:
                response_text = (
                    "Something went wrong reaching the assistant backend. "
                    "Please check that GROQ_API_KEY is set correctly."
                )
                intent = INTENT_OFF_TOPIC
                confidence = 0.0
                st.error(f"Backend error: {e}")
            elapsed_ms = round((time.time() - start) * 1000)

        meta = {
            "intent": intent,
            "confidence": confidence,
            "response_time_ms": elapsed_ms,
            "query": query,
        }

        style = INTENT_STYLE[intent]
        st.markdown(
            f'<span class="intent-badge" style="color:{style["color"]}; '
            f'background:{style["bg"]};">{style["label"]}</span>'
            f'<span class="meta-line">{elapsed_ms} ms · confidence {confidence:.2f}</span>',
            unsafe_allow_html=True,
        )

        if intent == INTENT_ESCALATE:
            st.markdown(
                f'<div class="escalation-box"><strong>Connecting to a human agent</strong>'
                f'<br>{response_text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.write(response_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "meta": meta,
    })
    st.rerun()
