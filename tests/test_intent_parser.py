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
