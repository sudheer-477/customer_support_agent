"""
care_intelligence_flow.py — Care Intelligence support agent (CrewAI + Groq)
------------------------------------------------------------------------------
Combines intent detection and RAG retrieval into one branching flow:

    classify_intent (keyword pre-filter -> LLM fallback)
            |
     route_intent  <-- confidence guardrail applied here
       /   |   \\    \\
    faq complaint escalate off_topic
     |       |        |         |
  RAG      canned    canned    canned
  answer  response   "connecting  polite
                      to human"   refusal

The `intent` value lives on `SupportState.intent` as its own field for the
entire run — every branch reads it, nothing recomputes or overloads it.

Install:
    pip install crewai sentence-transformers faiss-cpu numpy pydantic --break-system-packages

Setup:
    export GROQ_API_KEY="gsk_..."

Run:
    python care_intelligence_flow.py "Do you integrate with Epic?"
"""

import os
import sys

from crewai import LLM
from crewai.flow.flow import Flow, start, router, listen

from agent_core import (
    SupportState,
    INTENT_FAQ, INTENT_COMPLAINT, INTENT_ESCALATE, INTENT_OFF_TOPIC,
    keyword_preclassify, parse_intent_response, INTENT_CLASSIFIER_PROMPT,
    apply_confidence_guardrail, apply_relevance_guardrail,
    OFF_TOPIC_RESPONSE, ESCALATE_RESPONSE,
    COMPLAINT_RESPONSE_TEMPLATE, COMPLAINT_RESPONSE_ABUSIVE_TEMPLATE,
)
from rag_retriever import FAQRetriever


GROQ_MODEL = "groq/llama-3.3-70b-versatile"  # swap for any Groq-hosted model


class CareIntelligenceSupportFlow(Flow[SupportState]):
    """One run of this flow == one customer message handled end to end."""

    def __init__(self, query: str):
        super().__init__()
        self.state.query = query
        self.llm = LLM(model=GROQ_MODEL, api_key=os.environ.get("GROQ_API_KEY"), temperature=0.2)
        self.retriever = FAQRetriever()

    # -- Step 1: intent detection --------------------------------------
    @start()
    def classify_intent(self):
        pre = keyword_preclassify(self.state.query)
        if pre is not None:
            result = pre
        else:
            prompt = INTENT_CLASSIFIER_PROMPT.format(query=self.state.query)
            raw = self.llm.call(messages=[{"role": "user", "content": prompt}])
            result = parse_intent_response(raw)

        self.state.intent = result["intent"]
        self.state.confidence = result["confidence"]
        self.state.abusive = result.get("abusive", False)
        return self.state

    # -- Step 2: routing (guardrails applied here) -----------------------
    @router(classify_intent)
    def route_intent(self):
        # Guardrail: low-confidence classification always escalates,
        # regardless of which intent it guessed.
        routed = apply_confidence_guardrail(self.state.intent, self.state.confidence)
        self.state.intent = routed
        return routed  # "faq" | "complaint" | "escalate" | "off_topic"

    # -- Branch: FAQ -> RAG retrieval + grounded generation ---------------
    @listen(INTENT_FAQ)
    def handle_faq(self):
        chunks = self.retriever.search(self.state.query, top_k=3)
        top_score = chunks[0].score if chunks else 0.0

        # Guardrail: retrieval relevance check. If nothing relevant was
        # found, don't let the LLM free-associate an answer — treat it
        # as off_topic instead.
        effective_intent = apply_relevance_guardrail(INTENT_FAQ, top_score)
        if effective_intent == INTENT_OFF_TOPIC:
            self.state.intent = INTENT_OFF_TOPIC
            self.state.response = OFF_TOPIC_RESPONSE
            return self.state

        self.state.retrieved_context = self.retriever.format_context(chunks)
        self.state.retrieval_score = top_score

        system_prompt = (
            "You are the Care Intelligence support assistant. Answer using "
            "ONLY the provided FAQ context. Be concise and accurate. Never "
            "invent facts not present in the context."
        )
        user_prompt = (
            f"Context:\n{self.state.retrieved_context}\n\n"
            f"Customer question: {self.state.query}\n\n"
            "Answer using only the context above."
        )
        answer = self.llm.call(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        self.state.response = answer
        return self.state

    # -- Branch: complaint -> empathetic response, different from FAQ tone
    @listen(INTENT_COMPLAINT)
    def handle_complaint(self):
        self.state.response = (
            COMPLAINT_RESPONSE_ABUSIVE_TEMPLATE if self.state.abusive
            else COMPLAINT_RESPONSE_TEMPLATE
        )
        if self.state.abusive:
            self.state.escalated = True
        return self.state

    # -- Branch: escalate -> fixed handoff message ------------------------
    @listen(INTENT_ESCALATE)
    def handle_escalate(self):
        self.state.escalated = True
        self.state.response = ESCALATE_RESPONSE
        return self.state

    # -- Branch: off-topic -> polite refusal, no LLM call needed ----------
    @listen(INTENT_OFF_TOPIC)
    def handle_off_topic(self):
        self.state.response = OFF_TOPIC_RESPONSE
        return self.state


def run(query: str) -> SupportState:
    flow = CareIntelligenceSupportFlow(query)
    flow.kickoff()
    return flow.state


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Do you integrate with Epic?"
    final_state = run(query)
    print(f"\nQuery:      {final_state.query}")
    print(f"Intent:     {final_state.intent}  (confidence {final_state.confidence:.2f})")
    print(f"Escalated:  {final_state.escalated}")
    print(f"Response:   {final_state.response}")
