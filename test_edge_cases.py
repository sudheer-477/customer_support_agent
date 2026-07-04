"""
test_edge_cases.py — guardrail + routing tests, no live LLM required
----------------------------------------------------------------------
Exercises agent_core.py's guardrail functions directly against a set of
adversarial and edge-case inputs: rude language, gibberish, off-topic
questions, legitimate FAQ questions, complaints, and explicit escalation
requests.

This does NOT call Groq — it tests the deterministic guardrail layer
(keyword pre-filter, confidence override, relevance override) that runs
independently of the LLM. This is exactly the layer that must hold up
even if the LLM misbehaves, so it's tested in isolation.

Run:
    python test_edge_cases.py
"""

from agent_core import (
    INTENT_FAQ, INTENT_COMPLAINT, INTENT_ESCALATE, INTENT_OFF_TOPIC,
    keyword_preclassify, parse_intent_response,
    apply_confidence_guardrail, apply_relevance_guardrail,
    is_gibberish, detect_abusive,
)


def mock_llm_classify(query: str) -> dict:
    """Stand-in for the real Groq call — a tiny rule-based mock so this
    test suite runs with zero external dependencies. Deliberately dumber
    than the real LLM so we can see the guardrails, not the model, doing
    the work in these tests."""
    lowered = query.lower()
    if "epic" in lowered or "hipaa" in lowered or "pricing" in lowered or "trial" in lowered:
        return {"intent": INTENT_FAQ, "confidence": 0.9, "abusive": False}
    if "weather" in lowered or "capital of" in lowered:
        return {"intent": INTENT_OFF_TOPIC, "confidence": 0.85, "abusive": False}
    # ambiguous / unclear input -> deliberately low confidence
    return {"intent": INTENT_FAQ, "confidence": 0.3, "abusive": False}


def classify(query: str) -> dict:
    """Mirrors the real flow's classify_intent + route_intent steps."""
    pre = keyword_preclassify(query)
    result = pre if pre is not None else mock_llm_classify(query)
    routed_intent = apply_confidence_guardrail(result["intent"], result["confidence"])
    return {
        "intent": routed_intent,
        "confidence": result["confidence"],
        "abusive": result.get("abusive", False),
        "pre_filtered": pre is not None,
    }


TEST_CASES = [
    # (query, expected_intent, description)
    ("You guys are useless idiots, this app is garbage", INTENT_COMPLAINT, "rude/abusive input"),
    ("asdkjfh qwerqwer zzzzxxxx", INTENT_OFF_TOPIC, "gibberish"),
    ("What's the weather like today?", INTENT_OFF_TOPIC, "off-topic (but not gibberish)"),
    ("Can you integrate with Epic?", INTENT_FAQ, "legitimate FAQ question"),
    ("Is Care Intelligence HIPAA compliant?", INTENT_FAQ, "legitimate FAQ question"),
    ("My CareBot crashed and lost my patient notes", INTENT_COMPLAINT, "genuine complaint"),
    ("I want to talk to a human agent right now", INTENT_ESCALATE, "explicit escalation request"),
    ("hm ok maybe idk", INTENT_ESCALATE, "ambiguous/low-confidence -> forced escalate"),
    ("blah", INTENT_ESCALATE, "ambiguous single word, not garbled -> low confidence -> escalate"),
]


def run_tests():
    print(f"{'Query':<50} {'Expected':<12} {'Got':<12} {'Conf':<6} {'Result'}")
    print("-" * 100)
    passed, failed = 0, 0
    for query, expected, description in TEST_CASES:
        result = classify(query)
        ok = result["intent"] == expected
        passed += ok
        failed += not ok
        status = "PASS" if ok else "FAIL"
        display_query = (query[:47] + "...") if len(query) > 50 else query
        print(f"{display_query:<50} {expected:<12} {result['intent']:<12} "
              f"{result['confidence']:<6.2f} {status}  ({description})")

    print("-" * 100)
    print(f"{passed} passed, {failed} failed out of {len(TEST_CASES)}")

    # Unit checks on the primitives directly
    print("\nPrimitive guardrail checks:")
    assert is_gibberish("asdkjfh qwerqwer") is True
    print("  is_gibberish('asdkjfh qwerqwer') -> True [ok]")
    assert is_gibberish("What is your pricing model?") is False
    print("  is_gibberish('What is your pricing model?') -> False [ok]")
    assert detect_abusive("you are useless and stupid") is True
    print("  detect_abusive('you are useless and stupid') -> True [ok]")
    assert apply_confidence_guardrail(INTENT_FAQ, 0.3) == INTENT_ESCALATE
    print("  apply_confidence_guardrail(faq, 0.3) -> escalate [ok]")
    assert apply_relevance_guardrail(INTENT_FAQ, 0.2) == INTENT_OFF_TOPIC
    print("  apply_relevance_guardrail(faq, low_score) -> off_topic [ok]")
    assert parse_intent_response("not json at all")["intent"] == INTENT_OFF_TOPIC
    print("  parse_intent_response(garbage) -> off_topic, low confidence [ok]")

    print("\nAll primitive checks passed.")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
