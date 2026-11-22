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

ActionType = Literal["FETCH", "FOLLOW", "GO_TO", "STOP", "PATROL", "SIT", "QUACK", "SKILL"]
UrgencyType = Literal["LOW", "MEDIUM", "HIGH"]

VALID_ACTIONS: set[str] = {"FETCH", "FOLLOW", "GO_TO", "STOP", "PATROL", "SIT", "QUACK", "SKILL"}
VALID_URGENCIES: set[str] = {"LOW", "MEDIUM", "HIGH"}

SKILL_ALIASES: dict[str, str] = {
    "bow": "bow",
    "take a bow": "bow",
    "kick": "kick",
    "wave": "wave",
    "byebye": "byebye",
    "bye-bye": "byebye",
    "bye bye": "byebye",
    "call": "call",
    "karate": "karate_kicks",
    "karate kicks": "karate_kicks",
    "jazzwalk": "jazzwalk",
    "jazz walk": "jazzwalk",
    "ground pick": "ground_pick",
    "pickup": "ground_pick",
}


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

    # Emergency stop check (highest priority)
    if re.search(r"\b(stop|halt|freeze)\b", lower):
        return ParsedIntent(action="STOP", urgency="HIGH")

    # Vocalization check
    if re.search(r"\b(quack|chirp|speak)\b", lower):
        return ParsedIntent(action="QUACK", target="quack", urgency="LOW")

    # Expressive / episodic motor skills
    for phrase, canonical in sorted(SKILL_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", lower):
            return ParsedIntent(action="SKILL", target=canonical, urgency="MEDIUM")

    if "fetch" in lower or "bring" in lower or "get" in lower:
        # Extract target object
        match = re.search(r"\b(ball|marker|stick|toy|bone|object)\b", lower)
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


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from a model generation string."""
    raw = raw_output.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model output")
    parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Decoded JSON payload must be an object")
    return parsed


def parse_voice_command_stateless(
    model: Any,
    tokenizer: Any,
    raw_audio_text: str,
) -> ParsedIntent:
    """
    One-shot execution: Input string -> Typed Intent.
    If model is provided, invokes it statelessly (callable or `.generate()` with tokenizer)
    and validates the JSON output. Falls back cleanly to deterministic parsing if no model
    is supplied or if generation/validation fails.
    """
    if model is None:
        return parse_intent_fallback(raw_audio_text)

    prompt = (
        "Extract structured intent from command: '" + raw_audio_text + "'\n"
        "Output JSON only adhering to schema: "
        '{"action": "FETCH"|"FOLLOW"|"GO_TO"|"STOP"|"PATROL"|"SIT"|"QUACK"|"SKILL", '
        '"target": string|null, "urgency": "LOW"|"MEDIUM"|"HIGH"}\nJSON:'
    )

    try:
        if hasattr(model, "generate"):
            if tokenizer is None:
                raw_out = model.generate(prompt)
            else:
                encoded = tokenizer(prompt)
                if isinstance(encoded, dict):
                    generated = model.generate(**encoded)
                else:
                    generated = model.generate(encoded)
                if hasattr(tokenizer, "decode") and not isinstance(generated, str):
                    tokens = generated[0] if isinstance(generated, (list, tuple)) else generated
                    raw_out = tokenizer.decode(tokens)
                else:
                    raw_out = generated
        elif callable(model):
            raw_out = model(prompt)
        else:
            return parse_intent_fallback(raw_audio_text)

        if isinstance(raw_out, dict):
            data = raw_out
        elif isinstance(raw_out, str):
            data = _extract_json_object(raw_out)
        else:
            return parse_intent_fallback(raw_audio_text)

        return ParsedIntent.from_dict(data)
    except Exception:
        return parse_intent_fallback(raw_audio_text)


class IntentParser:
    """Convenience wrapper for extracting structured intents."""

    def __init__(self, model: Any = None, tokenizer: Any = None) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def parse(self, text: str) -> dict[str, Any]:
        """Extracts intent and returns a dictionary with intent, target, and urgency."""
        parsed = parse_voice_command_stateless(self.model, self.tokenizer, text)
        return {
            "intent": parsed.action,
            "target": parsed.target,
            "urgency": parsed.urgency,
        }

