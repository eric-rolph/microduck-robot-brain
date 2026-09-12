"""
Executive Behavior Tree engine for Tier 3.
Implements reactive sequences, fallback selectors, asymmetric parallel
composition for ambient gesture blending, and stability-subordinated expressions.
"""

from __future__ import annotations
import math
from typing import Any, Callable, List, Mapping, Optional
from .skill_latch import SkillStatus


class Blackboard:
    """Shared state store for cross-node context, task metrics, and retry tracking."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "retry_counts": {},
            "last_failure_reason": None,
            "published_commands": [],
            "executed_gestures": [],
        }

    def increment_failure(self, task_key: str) -> int:
        count = self.data["retry_counts"].get(task_key, 0) + 1
        self.data["retry_counts"][task_key] = count
        return count

    def get_failure_count(self, task_key: str) -> int:
        return self.data["retry_counts"].get(task_key, 0)

    def reset_failure(self, task_key: str) -> None:
        self.data["retry_counts"][task_key] = 0

    def record_gesture(self, gesture_id: str) -> None:
        self.data["executed_gestures"].append(gesture_id)


class BehaviorNode:
    """Base behavior tree node."""

    def __init__(self, name: str, blackboard: Optional[Blackboard] = None) -> None:
        self.name = name
        self.blackboard: Blackboard = blackboard if blackboard is not None else Blackboard()

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        raise NotImplementedError("Subclasses must implement tick()")

    def abort(self) -> None:
        """Called when a node is preempted by a parent composite."""
        pass


class SequenceNode(BehaviorNode):
    """
    Sequential composite node.
    Runs children in order until one returns RUNNING or FAILURE.
    Returns SUCCESS when all children succeed.
    """

    def __init__(
        self,
        name: str,
        blackboard: Optional[Blackboard],
        children: List[BehaviorNode],
    ) -> None:
        super().__init__(name, blackboard)
        self.children = children
        self.current_idx = 0

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        while self.current_idx < len(self.children):
            status = self.children[self.current_idx].tick(world_state)
            if status in (SkillStatus.RUNNING, SkillStatus.FAILURE):
                if status == SkillStatus.FAILURE:
                    self.current_idx = 0
                return status
            self.current_idx += 1

        self.current_idx = 0
        return SkillStatus.SUCCESS

    def abort(self) -> None:
        if self.current_idx < len(self.children):
            self.children[self.current_idx].abort()
        self.current_idx = 0


class SelectorNode(BehaviorNode):
    """
    Fallback composite node.
    Runs children in order until one returns RUNNING or SUCCESS.
    Advances to next child on FAILURE.
    """

    def __init__(
        self,
        name: str,
        blackboard: Optional[Blackboard],
        children: List[BehaviorNode],
    ) -> None:
        super().__init__(name, blackboard)
        self.children = children
        self.current_idx = 0

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        while self.current_idx < len(self.children):
            status = self.children[self.current_idx].tick(world_state)
            if status in (SkillStatus.RUNNING, SkillStatus.SUCCESS):
                if status == SkillStatus.SUCCESS:
                    self.current_idx = 0
                return status
            self.current_idx += 1

        self.current_idx = 0
        return SkillStatus.FAILURE

    def abort(self) -> None:
        if self.current_idx < len(self.children):
            self.children[self.current_idx].abort()
        self.current_idx = 0


class AsymmetricParallelNode(BehaviorNode):
    """
    Ticks all children concurrently each cycle.
    Completion is strictly anchored to primary_idx.
    Secondary (ambient) children are aborted when the primary child completes or fails.
    """

    def __init__(
        self,
        name: str,
        blackboard: Optional[Blackboard],
        children: List[BehaviorNode],
        primary_idx: int = 0,
    ) -> None:
        super().__init__(name, blackboard)
        self.children = children
        self.primary_idx = primary_idx

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        primary_status: Optional[SkillStatus] = None

        for idx, child in enumerate(self.children):
            status = child.tick(world_state)
            if idx == self.primary_idx:
                primary_status = status

        if primary_status in (SkillStatus.SUCCESS, SkillStatus.FAILURE):
            for idx, child in enumerate(self.children):
                if idx != self.primary_idx:
                    child.abort()
            return primary_status

        return SkillStatus.RUNNING

    def abort(self) -> None:
        for child in self.children:
            child.abort()


class SafeExpressionNode(BehaviorNode):
    """
    Executes non-locomotive gestures only when physical stability allows.
    Strictly subordinate to balance and low-level safety limits.
    If stability != 'HIGH' or roughness == 'HIGH', suppresses gesture and yields SUCCESS safely.
    """

    def __init__(
        self,
        name: str,
        blackboard: Optional[Blackboard],
        gesture_id: str,
        duration_ticks: int = 10,
        gesture_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(name, blackboard)
        self.gesture_id = gesture_id
        self.duration_ticks = duration_ticks
        self.elapsed_ticks = 0
        self.gesture_callback = gesture_callback

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        stability = str(world_state.get("stability", "HIGH")).upper()
        roughness = str(world_state.get("roughness", "LOW")).upper()

        # Balance rule: Expressions are strictly subordinate to stability
        if stability != "HIGH" or roughness == "HIGH":
            self.elapsed_ticks = 0
            return SkillStatus.SUCCESS

        # Execute gesture trigger on first active tick
        if self.elapsed_ticks == 0:
            self.blackboard.record_gesture(self.gesture_id)
            if self.gesture_callback is not None:
                self.gesture_callback(self.gesture_id)

        self.elapsed_ticks += 1
        if self.elapsed_ticks >= self.duration_ticks:
            self.elapsed_ticks = 0
            return SkillStatus.SUCCESS

        return SkillStatus.RUNNING

    def abort(self) -> None:
        self.elapsed_ticks = 0


class AmbientBodyLanguageNode(BehaviorNode):
    """
    Generates sinusoidal head yaw and spine trim offsets.
    Zeroes out immediately if gait stability drops or roughness spikes.
    """

    def __init__(
        self,
        name: str,
        blackboard: Optional[Blackboard] = None,
        frequency_hz: float = 1.0,
        max_yaw_deg: float = 15.0,
        offset_callback: Optional[Callable[[float, float], None]] = None,
    ) -> None:
        super().__init__(name, blackboard)
        self.freq = frequency_hz
        self.max_yaw = max_yaw_deg
        self.tick_counter = 0
        self.offset_callback = offset_callback

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        stability = str(world_state.get("stability", "HIGH")).upper()
        roughness = str(world_state.get("roughness", "LOW")).upper()

        if stability != "HIGH" or roughness == "HIGH":
            if self.offset_callback is not None:
                self.offset_callback(0.0, 0.0)
            return SkillStatus.RUNNING

        t = self.tick_counter * 0.05  # Assumes 20 Hz BT tick rate
        yaw_offset = self.max_yaw * math.sin(2 * math.pi * self.freq * t)
        spine_trim = (self.max_yaw * 0.2) * math.cos(2 * math.pi * self.freq * t)

        if self.offset_callback is not None:
            self.offset_callback(yaw_offset, spine_trim)

        self.tick_counter += 1
        return SkillStatus.RUNNING

    def abort(self) -> None:
        self.tick_counter = 0
        if self.offset_callback is not None:
            self.offset_callback(0.0, 0.0)


class FailureExpressionBranch(BehaviorNode):
    """
    Selects between inquisitive head tilt (1-2 failures) and
    resigned head shake (3+ failures) using blackboard metrics.
    """

    def __init__(
        self,
        name: str,
        blackboard: Blackboard,
        monitored_task: str,
        gesture_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(name, blackboard)
        self.monitored_task = monitored_task
        self.tilt_expression = SafeExpressionNode(
            "HeadTilt",
            blackboard,
            gesture_id="HEAD_TILT",
            duration_ticks=5,
            gesture_callback=gesture_callback,
        )
        self.shake_expression = SafeExpressionNode(
            "HeadShake",
            blackboard,
            gesture_id="HEAD_SHAKE",
            duration_ticks=12,
            gesture_callback=gesture_callback,
        )

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        failure_count = self.blackboard.increment_failure(self.monitored_task)
        if failure_count >= 3:
            return self.shake_expression.tick(world_state)
        return self.tilt_expression.tick(world_state)

    def abort(self) -> None:
        self.tilt_expression.abort()
        self.shake_expression.abort()
