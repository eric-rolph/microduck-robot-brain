"""
Perception and WorldState sanitization for Tier 2.
Implements dual-threshold Schmitt triggers and temporal debouncers
to eliminate sensor noise and token flickering.
"""

from __future__ import annotations
from collections import deque
from typing import Literal

StateLevel = Literal["LOW", "MEDIUM", "HIGH"]


class SchmittTrigger:
    """
    Dual-threshold hysteresis filter for continuous signals.
    Prevents token flip-flopping at boundaries.
    """

    def __init__(
        self,
        low_threshold: float,
        high_threshold: float,
        initial_state: StateLevel = "LOW",
    ):
        if low_threshold >= high_threshold:
            raise ValueError(
                f"low_threshold ({low_threshold}) must be strictly less than high_threshold ({high_threshold})"
            )
        self.low_th = low_threshold
        self.high_th = high_threshold
        self.state: StateLevel = initial_state

    def update(self, measurement: float) -> StateLevel:
        if self.state == "LOW" and measurement >= self.high_th:
            self.state = "HIGH"
        elif self.state == "HIGH" and measurement <= self.low_th:
            self.state = "LOW"
        return self.state


class MultiLevelSchmittTrigger:
    """
    Three-state hysteresis filter (LOW, MEDIUM, HIGH) for stability and roughness.
    """

    def __init__(
        self,
        low_to_med: float = 0.35,
        med_to_low: float = 0.25,
        med_to_high: float = 0.65,
        high_to_med: float = 0.50,
        initial_state: StateLevel = "LOW",
    ):
        self.low_to_med = low_to_med
        self.med_to_low = med_to_low
        self.med_to_high = med_to_high
        self.high_to_med = high_to_med
        self.state: StateLevel = initial_state

    def update(self, measurement: float) -> StateLevel:
        if self.state == "LOW":
            if measurement >= self.med_to_high:
                self.state = "HIGH"
            elif measurement >= self.low_to_med:
                self.state = "MEDIUM"
        elif self.state == "MEDIUM":
            if measurement >= self.med_to_high:
                self.state = "HIGH"
            elif measurement <= self.med_to_low:
                self.state = "LOW"
        elif self.state == "HIGH":
            if measurement <= self.med_to_low:
                self.state = "LOW"
            elif measurement <= self.high_to_med:
                self.state = "MEDIUM"
        return self.state


class TemporalDebouncer:
    """
    Sliding-window majority filter for noisy boolean or detection signals.
    Requires state persistence across a rolling window before updating output.
    """

    def __init__(self, window_size: int = 5, required_ratio: float = 0.8):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not (0.0 <= required_ratio <= 1.0):
            raise ValueError("required_ratio must be between 0.0 and 1.0")

        self.window: deque[bool] = deque(maxlen=window_size)
        self.required_ratio = required_ratio

    def update(self, raw_detection: bool) -> bool:
        self.window.append(bool(raw_detection))
        if len(self.window) < self.window.maxlen:
            # During warmup, evaluate over available samples
            ratio = sum(self.window) / len(self.window)
            return ratio >= self.required_ratio
        ratio = sum(self.window) / len(self.window)
        return ratio >= self.required_ratio

    def reset(self) -> None:
        self.window.clear()
