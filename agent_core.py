"""
agent_core.py — Care Intelligence support agent core logic
------------------------------------------------------------
Intent classification, guardrails, and routing rules live here as pure
functions so they can be unit tested WITHOUT calling Groq. The CrewAI
Flow (care_intelligence_flow.py) imports these and wires them to a real
LLM. Keeping this separate is deliberate: guardrail behavior should be
verifiable in CI without needing a live API key.

The `intent` field is always kept as its own explicit variable on the
flow state (see SupportState below) — it is never bundled into the
response text or inferred implicitly. Every routing decision reads
from this one field.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional, List
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Intent vocabulary — closed set. This is guardrail #1: the model is never
# allowed to invent a new intent category. Anything that doesn't cleanly
# fit is forced into "off_topic" or "escalate".
# ---------------------------------------------------------------------------

INTENT_FAQ = "faq"
INTENT_COMPLAINT = "complaint"
INTENT_ESCALATE = "escalate"
INTENT_OFF_TOPIC = "off_topic"

VALID_INTENTS = {INTENT_FAQ, INTENT_COMPLAINT, INTENT_ESCALATE, INTENT_OFF_TOPIC}

# ---------------------------------------------------------------------------
# Guardrail thresholds
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.55     # below this -> force escalate, regardless of intent
RELEVANCE_THRESHOLD = 1.0       # below this retrieval score -> treat FAQ as off_topic
ABUSIVE_KEYWORDS = [
    "idiot", "stupid", "useless", "garbage", "trash", "shut up",
    "dumb", "hate you", "worthless", "pathetic",
]
ESCALATE_KEYWORDS = [
    "human", "agent", "representative", "real person",
    "speak to someone", "talk to a person", "manager",
]
COMPLAINT_KEYWORDS = [
    "broken", "crashed", "not working", "doesn't work", "refund",
    "cancel", "furious", "disappointed", "angry", "frustrated",
    "lost my data", "terrible", "worst",
]


# ---------------------------------------------------------------------------
# Flow state — `intent` is deliberately its own top-level field, separate
# from `response`, `retrieved_context`, etc. Nothing downstream mutates it
# except the router.
# ---------------------------------------------------------------------------

class SupportState(BaseModel):
    query: str = ""
    intent: str = ""            # <-- the separate intent variable
    confidence: float = 0.0
    abusive: bool = False
    retrieved_context: str = ""
    retrieval_score: float = 0.0
    response: str = ""
    escalated: bool = False


# ---------------------------------------------------------------------------
# Guardrail: pre-filter for abusive language.
# Runs BEFORE the LLM call so rude input can't be reasoned around by a
# jailbreak-style prompt hiding inside the message.
# ---------------------------------------------------------------------------

def detect_abusive(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ABUSIVE_KEYWORDS)


# ---------------------------------------------------------------------------
# Guardrail: gibberish detector.
# Cheap heuristic — low vowel ratio + no recognizable word + short/garbled.
# Gibberish is routed to off_topic rather than sent to the LLM as if it
# were a real question.
# ---------------------------------------------------------------------------

def is_gibberish(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) < 2:
        return True
    letters = re.sub(r"[^a-zA-Z]", "", cleaned)
    if not letters:
        return True
    vowels = re.findall(r"[aeiouAEIOU]", letters)
    vowel_ratio = len(vowels) / len(letters)
    # long consonant runs (e.g. "sdkjfh") are a strong garbled-keyboard signal
    has_long_consonant_run = re.search(r"[^aeiouAEIOU\s]{5,}", cleaned) is not None
    return vowel_ratio < 0.25 or has_long_consonant_run


# ---------------------------------------------------------------------------
# Keyword pre-routing (guardrail layer that runs alongside/instead of the
# LLM classifier for cheap, unambiguous cases). The LLM classifier is still
# the fallback for anything these don't catch.
# ---------------------------------------------------------------------------

def keyword_preclassify(text: str) -> Optional[dict]:
    lowered = text.lower()

    if is_gibberish(text):
        # Confident classification (deterministic heuristic), not a shaky
        # guess — so it should NOT be caught by the confidence-escalate
        # guardrail below. It's confidently off-topic, not ambiguous.
        return {"intent": INTENT_OFF_TOPIC, "confidence": 0.7, "abusive": False}

    if detect_abusive(text):
        return {"intent": INTENT_COMPLAINT, "confidence": 0.85, "abusive": True}

    if any(kw in lowered for kw in ESCALATE_KEYWORDS):
        return {"intent": INTENT_ESCALATE, "confidence": 0.9, "abusive": False}

    if any(kw in lowered for kw in COMPLAINT_KEYWORDS):
        return {"intent": INTENT_COMPLAINT, "confidence": 0.8, "abusive": False}

    return None  # fall through to LLM classification


# ---------------------------------------------------------------------------
# Prompt for the LLM-based intent classifier (used when keyword
# pre-classification doesn't confidently resolve the intent).
# ---------------------------------------------------------------------------

INTENT_CLASSIFIER_PROMPT = """Classify the customer message into exactly one \
of these intents: faq, complaint, escalate, off_topic.

- faq: a question about Care Intelligence's products, pricing, security, \
integrations, or support (answerable from a healthcare-AI-vendor FAQ).
- complaint: the customer is unhappy, reporting a problem, or expressing \
frustration with the product or service.
- escalate: the customer explicitly asks for a human, agent, or manager.
- off_topic: anything unrelated to Care Intelligence's products (weather, \
general knowledge, small talk, gibberish, or unrelated companies).

Respond with ONLY a JSON object, no other text:
{{"intent": "<one of faq|complaint|escalate|off_topic>", "confidence": <0.0-1.0>}}

Customer message: "{query}"
"""


def parse_intent_response(raw: str) -> dict:
    """Parse the LLM's JSON response. On any failure, default to a low
    confidence off_topic classification — the confidence guardrail will
    then force an escalation rather than let a malformed response silently
    proceed as if it were a confident answer."""
    try:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        payload = json.loads(match.group(0) if match else raw)
        intent = payload.get("intent", INTENT_OFF_TOPIC)
        confidence = float(payload.get("confidence", 0.0))
        if intent not in VALID_INTENTS:
            intent = INTENT_OFF_TOPIC
            confidence = min(confidence, 0.3)
        return {"intent": intent, "confidence": confidence, "abusive": False}
    except Exception:
        return {"intent": INTENT_OFF_TOPIC, "confidence": 0.0, "abusive": False}


# ---------------------------------------------------------------------------
# Guardrail: confidence-based escalation.
# Applied AFTER classification (keyword or LLM), overriding intent if the
# model isn't confident enough. This is the safety net for ambiguous input.
# ---------------------------------------------------------------------------

def apply_confidence_guardrail(intent: str, confidence: float) -> str:
    if confidence < CONFIDENCE_THRESHOLD:
        return INTENT_ESCALATE
    return intent


# ---------------------------------------------------------------------------
# Guardrail: RAG relevance check.
# Even if intent == "faq", if retrieval didn't find anything relevant, we
# do not let the LLM generate a free-floating answer — we treat it as
# off_topic instead. This prevents hallucinated answers to in-domain-
# sounding but unsupported questions.
# ---------------------------------------------------------------------------

def apply_relevance_guardrail(intent: str, top_retrieval_score: float) -> str:
    if intent == INTENT_FAQ and top_retrieval_score < RELEVANCE_THRESHOLD:
        return INTENT_OFF_TOPIC
    return intent


# ---------------------------------------------------------------------------
# Canned / templated responses for non-FAQ branches. Kept deterministic
# (not LLM-generated) so the tone and legal safety of complaint/escalation/
# off-topic messaging can't drift.
# ---------------------------------------------------------------------------

OFF_TOPIC_RESPONSE = (
    "I'm the Care Intelligence support assistant, so I can only help with "
    "questions about our products for healthcare centers (like CareScribe, "
    "CareTriage, CareBot, and CareInsights), pricing, security, or "
    "integrations. Could you ask something along those lines?"
)

ESCALATE_RESPONSE = (
    "Connecting you to a human agent now. Please hold — a member of our "
    "support team will join this conversation shortly."
)

COMPLAINT_RESPONSE_TEMPLATE = (
    "I'm sorry to hear you're running into this — that's frustrating, and "
    "I want to help get it resolved. I've logged your message so our "
    "support team has the details. Would you like me to connect you with "
    "a human agent now, or can I try to help directly first?"
)

COMPLAINT_RESPONSE_ABUSIVE_TEMPLATE = (
    "I hear that you're frustrated, and I'm sorry for the trouble. I want "
    "to get this fixed for you. I'm connecting you with a human agent who "
    "can look into this directly."
)
