"""
Unit tests for stateless intent parser.
"""

import pytest
from microduck_brain.intent_parser import (
    ParsedIntent,
    parse_intent_fallback,
    parse_voice_command_stateless,
)


def test_fetch_intent():
    intent = parse_intent_fallback("Ducky, fetch the ball!")
    assert intent.action == "FETCH"
    assert intent.target == "ball"
    assert intent.urgency == "MEDIUM"


def test_emergency_stop():
    intent = parse_intent_fallback("Stop immediately!")
    assert intent.action == "STOP"
    assert intent.urgency == "HIGH"


def test_follow_intent():
    intent = parse_intent_fallback("Come here, hurry!")
    assert intent.action == "FOLLOW"
    assert intent.target == "person"
    assert intent.urgency == "HIGH"


def test_sit_intent():
    intent = parse_intent_fallback("Sit down and rest.")
    assert intent.action == "SIT"
    assert intent.urgency == "LOW"


def test_parsed_intent_validation():
    valid = ParsedIntent(action="FETCH", target="ball", urgency="HIGH")
    d = valid.to_dict()
    assert d["action"] == "FETCH"

    reconstructed = ParsedIntent.from_dict(d)
    assert reconstructed == valid

    with pytest.raises(ValueError):
        ParsedIntent.from_dict({"action": "INVALID_ACTION"})


def test_parse_voice_command_stateless():
    intent = parse_voice_command_stateless(None, None, "Ducky, bring the toy")
    assert intent.action == "FETCH"
    assert intent.target == "toy"


def test_quack_and_skill_intents():
    q = parse_intent_fallback("Ducky, quack for me!")
    assert q.action == "QUACK"
    assert q.target == "quack"

    bow = parse_intent_fallback("Take a bow")
    assert bow.action == "SKILL"
    assert bow.target == "bow"

    kick = parse_intent_fallback("Kick the ball")
    assert kick.action == "SKILL"
    assert kick.target == "kick"

    wave = parse_intent_fallback("Wave hello")
    assert wave.action == "SKILL"
    assert wave.target == "wave"

    byebye = parse_intent_fallback("Say bye bye")
    assert byebye.action == "SKILL"
    assert byebye.target == "byebye"

    karate = parse_intent_fallback("Do karate kicks")
    assert karate.action == "SKILL"
    assert karate.target == "karate_kicks"

    jazz = parse_intent_fallback("Do the jazz walk")
    assert jazz.action == "SKILL"
    assert jazz.target == "jazzwalk"


def test_parse_voice_command_stateless_invokes_callable_and_hf_model():
    # 1. Callable model returning non-FETCH JSON proves hardcoded stub is gone
    called = []

    def fake_llm(prompt: str) -> str:
        called.append(prompt)
        return 'Sure! {"action": "SIT", "target": null, "urgency": "LOW"}'

    intent = parse_voice_command_stateless(fake_llm, None, "Please rest now")
    assert len(called) == 1
    assert intent == ParsedIntent(action="SIT", target=None, urgency="LOW")

    # 2. HuggingFace-style model.generate + tokenizer.decode
    class _FakeTokenizer:
        def __call__(self, prompt: str, return_tensors: str = "pt"):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, tokens, skip_special_tokens: bool = True) -> str:
            return '{"action": "SKILL", "target": "bow", "urgency": "MEDIUM"}'

    class _FakeModel:
        def generate(self, **kwargs):
            return [[1, 2, 3, 4, 5]]

    intent_hf = parse_voice_command_stateless(_FakeModel(), _FakeTokenizer(), "greet the guests")
    assert intent_hf == ParsedIntent(action="SKILL", target="bow", urgency="MEDIUM")

    # 3. Malformed model output falls back safely to deterministic parser
    intent_bad = parse_voice_command_stateless(lambda _p: "not valid json", None, "quack!")
    assert intent_bad.action == "QUACK"
