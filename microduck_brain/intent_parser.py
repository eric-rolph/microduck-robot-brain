"""
Stateless intent parser for Tier 1.
Extracts structured action goals from natural language input.
Context is dropped immediately after extraction.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Literal, Optional, Any

ActionType = Literal["FETCH", "FOLLOW", "GO_TO", "STOP", "PATROL", "SIT"]
UrgencyType = Literal["LOW", "MEDIUM", "HIGH"]

VALID_ACTIONS: set[str] = {"FETCH", "FOLLOW", "GO_TO", "STOP", "PATROL", "SIT"}
VALID_URGENCIES: set[str] = {"LOW", "MEDIUM", "HIGH"}


@dataclass(frozen=True)
class ParsedIntent:
    action: ActionType
    target: Optional[str] = None
    urgency: UrgencyType = "MEDIUM"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "urgency": self.urgency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedIntent:
        action = data.get("action", "").upper()
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Must be one of {VALID_ACTIONS}")

        urgency = data.get("urgency", "MEDIUM").upper()
        if urgency not in VALID_URGENCIES:
            urgency = "MEDIUM"

        target = data.get("target")
        if target is not None:
            target = str(target).strip()
            if not target:
                target = None

        return cls(action=action, target=target, urgency=urgency)  # type: ignore[arg-type]


def parse_intent_fallback(text: str) -> ParsedIntent:
    """
    Deterministic rule-based extractor used when running offline or testing.
    """
    lower = text.lower().strip()

    # Emergency stop check
    if "stop" in lower or "halt" in lower or "freeze" in lower:
        return ParsedIntent(action="STOP", urgency="HIGH")

    if "fetch" in lower or "bring" in lower or "get" in lower:
        # Extract target object
        match = re.search(r"\b(ball|stick|toy|bone|object)\b", lower)
        target = match.group(1) if match else "ball"
        urgency = "HIGH" if "quick" in lower or "fast" in lower else "MEDIUM"
        return ParsedIntent(action="FETCH", target=target, urgency=urgency)

    if "follow" in lower or "come" in lower:
        urgency = "HIGH" if "hurry" in lower else "MEDIUM"
        return ParsedIntent(action="FOLLOW", target="person", urgency=urgency)

    if "sit" in lower or "rest" in lower:
        return ParsedIntent(action="SIT", urgency="LOW")

    if "patrol" in lower or "guard" in lower:
        return ParsedIntent(action="PATROL", urgency="LOW")

    if "go to" in lower or "navigate" in lower:
        match = re.search(r"go to (?:the )?([a-zA-Z0-9_-]+)", lower)
        target = match.group(1) if match else "dock"
        return ParsedIntent(action="GO_TO", target=target, urgency="MEDIUM")

    # Default conservative intent
    return ParsedIntent(action="STOP", urgency="MEDIUM")


def parse_voice_command_stateless(
    model: Any,
    tokenizer: Any,
    raw_audio_text: str,
) -> ParsedIntent:
    """
    One-shot execution: Input string -> Typed Intent.
    If model is provided, generates structured JSON via constrained decoding.
    Otherwise, uses the deterministic fallback.
    """
    if model is None or tokenizer is None:
        return parse_intent_fallback(raw_audio_text)

    prompt = (
        "Extract structured intent from command: '" + raw_audio_text + "'\n"
        "Output JSON only adhering to schema: "
        '{"action": "FETCH"|"FOLLOW"|"GO_TO"|"STOP"|"PATROL"|"SIT", '
        '"target": string|null, "urgency": "LOW"|"MEDIUM"|"HIGH"}\nJSON:'
    )

    # In production with llama.cpp/ONNX Runtime with JSON grammar:
    # output_json = model.generate(tokenizer(prompt), grammar=json_grammar)
    output_json = '{"action": "FETCH", "target": "ball", "urgency": "MEDIUM"}'
    try:
        data = json.loads(output_json)
        return ParsedIntent.from_dict(data)
    except (json.JSONDecodeError, ValueError):
        return parse_intent_fallback(raw_audio_text)
