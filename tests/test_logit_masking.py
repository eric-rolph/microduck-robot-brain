"""
Unit tests for dynamic logit masking.
"""

import torch
from microduck_brain.logit_masking import mask_invalid_actions

TOKEN_MAP = {
    0: "STOP",
    1: "SEARCH",
    2: "APPROACH",
    3: "PICKUP",
    4: "RETURN",
    5: "NORMAL_PACE",
    6: "FAST_PACE",
}


def test_pickup_masked_when_ball_not_visible():
    logits = torch.tensor([1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0])
    world_state = {"ball_visible": False, "distance": 0.2, "stability": "HIGH"}

    masked = mask_invalid_actions(logits, TOKEN_MAP, world_state)
    assert masked[3] == float("-inf")
    assert masked[0] == 1.0


def test_pickup_masked_when_ball_far():
    logits = torch.tensor([1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0])
    world_state = {"ball_visible": True, "distance": 1.5, "stability": "HIGH"}

    masked = mask_invalid_actions(logits, TOKEN_MAP, world_state)
    assert masked[3] == float("-inf")
    # APPROACH should be allowed
    assert masked[2] == 1.0


def test_pickup_allowed_when_ball_close_and_visible():
    logits = torch.tensor([1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0])
    world_state = {"ball_visible": True, "distance": 0.15, "stability": "HIGH"}

    masked = mask_invalid_actions(logits, TOKEN_MAP, world_state)
    assert masked[3] == 5.0


def test_pace_masked_when_unstable():
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 4.0])
    world_state = {"ball_visible": True, "distance": 1.0, "stability": "LOW"}

    masked = mask_invalid_actions(logits, TOKEN_MAP, world_state)
    assert masked[5] == float("-inf")
    assert masked[6] == float("-inf")


def test_return_masked_without_possession():
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0])
    world_state = {"ball_visible": True, "ball_grasped": False}

    masked = mask_invalid_actions(logits, TOKEN_MAP, world_state)
    assert masked[4] == float("-inf")
