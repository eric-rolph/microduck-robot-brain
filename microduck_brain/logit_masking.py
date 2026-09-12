"""
Dynamic logit masking for precondition enforcement.
Sets the logits of physically invalid actions to -inf prior to sampling.
"""

from __future__ import annotations
from typing import Any, Mapping
import torch


def mask_invalid_actions(
    logits: torch.Tensor,
    token_to_action: Mapping[int, str],
    world_state: Mapping[str, Any],
    pickup_distance_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Sets logits of physically invalid actions to -inf based on current WorldState.

    Precondition rules:
    - PICKUP requires ball_visible=True and distance <= pickup_distance_threshold
    - APPROACH requires ball_visible=True
    - NORMAL_PACE and FAST_PACE are masked if stability == 'LOW'
    - RETURN requires target in possession (e.g. ball_grasped=True)
    """
    masked_logits = logits.clone()

    ball_visible = bool(world_state.get("ball_visible", False))
    distance = float(world_state.get("distance", float("inf")))
    stability = str(world_state.get("stability", "HIGH")).upper()
    ball_grasped = bool(world_state.get("ball_grasped", False))

    for token_id, action in token_to_action.items():
        action_upper = action.upper()

        if action_upper == "PICKUP":
            if not ball_visible or distance > pickup_distance_threshold:
                masked_logits[..., token_id] = float("-inf")

        elif action_upper == "APPROACH":
            if not ball_visible or distance <= pickup_distance_threshold:
                masked_logits[..., token_id] = float("-inf")

        elif action_upper in ("NORMAL_PACE", "FAST_PACE"):
            if stability == "LOW":
                masked_logits[..., token_id] = float("-inf")

        elif action_upper == "RETURN":
            if not ball_grasped:
                masked_logits[..., token_id] = float("-inf")

    return masked_logits
