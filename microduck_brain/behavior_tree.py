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


class ObstacleAvoidanceNode(BehaviorNode):
    """
    Reactive navigation node for dynamic obstacle circumnavigation.
    When ToF distance sensor detects a frontal obstacle, commands yaw
    deflection, forward bypass, and re-alignment to clear the obstacle.
    """

    def __init__(
        self,
        name: str,
        blackboard: Optional[Blackboard] = None,
        turn_ticks: int = 15,
        bypass_ticks: int = 25,
        realign_ticks: int = 15,
        velocity_callback: Optional[Callable[[float, float, float], None]] = None,
    ) -> None:
        super().__init__(name, blackboard)
        self.turn_ticks = turn_ticks
        self.bypass_ticks = bypass_ticks
        self.realign_ticks = realign_ticks
        self.velocity_callback = velocity_callback
        self.phase = 0  # 0: idle/check, 1: turn_away, 2: bypass, 3: realign
        self.step_in_phase = 0

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        obstacle = bool(world_state.get("obstacle_detected", False))

        if self.phase == 0:
            if not obstacle:
                return SkillStatus.SUCCESS
            # Start avoidance maneuver
            self.phase = 1
            self.step_in_phase = 0

        if self.phase == 1:
            # Turn left to clear frontal line
            if self.velocity_callback is not None:
                self.velocity_callback(0.02, 0.0, 0.45)
            self.step_in_phase += 1
            if self.step_in_phase >= self.turn_ticks:
                self.phase = 2
                self.step_in_phase = 0
            return SkillStatus.RUNNING

        elif self.phase == 2:
            # Advance along flank past obstacle
            if self.velocity_callback is not None:
                self.velocity_callback(0.08, 0.02, 0.0)
            self.step_in_phase += 1
            if self.step_in_phase >= self.bypass_ticks:
                self.phase = 3
                self.step_in_phase = 0
            return SkillStatus.RUNNING

        elif self.phase == 3:
            # Steer right to re-align heading with target
            if self.velocity_callback is not None:
                self.velocity_callback(0.03, 0.0, -0.45)
            self.step_in_phase += 1
            if self.step_in_phase >= self.realign_ticks:
                self.phase = 0
                self.step_in_phase = 0
                if self.velocity_callback is not None:
                    self.velocity_callback(0.0, 0.0, 0.0)
                return SkillStatus.SUCCESS
            return SkillStatus.RUNNING

        return SkillStatus.RUNNING

    def abort(self) -> None:
        self.phase = 0
        self.step_in_phase = 0
        if self.velocity_callback is not None:
            self.velocity_callback(0.0, 0.0, 0.0)


class BeakGraspActionNode(BehaviorNode):
    """
    Physical object grasping node with Microduck beak.
    Coordinates deep crouch, beak motor opening, contact lock, and standing recovery.
    """

    def __init__(
        self,
        name: str,
        blackboard: Optional[Blackboard] = None,
        crouch_ticks: int = 15,
        grasp_ticks: int = 10,
        rise_ticks: int = 15,
        beak_callback: Optional[Callable[[bool, bool], None]] = None,
    ) -> None:
        super().__init__(name, blackboard)
        self.crouch_ticks = crouch_ticks
        self.grasp_ticks = grasp_ticks
        self.rise_ticks = rise_ticks
        self.beak_callback = beak_callback
        self.phase = 0  # 0: crouch/open, 1: clamp/weld, 2: rise
        self.step_in_phase = 0

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        if self.phase == 0:
            # Crouching and opening beak
            if self.beak_callback is not None:
                self.beak_callback(True, False)  # beak_open=True, grasp_weld=False
            self.step_in_phase += 1
            if self.step_in_phase >= self.crouch_ticks:
                self.phase = 1
                self.step_in_phase = 0
            return SkillStatus.RUNNING

        elif self.phase == 1:
            # Clamping beak and locking physical equality weld
            if self.beak_callback is not None:
                self.beak_callback(False, True)  # beak_open=False, grasp_weld=True
            self.step_in_phase += 1
            if self.step_in_phase >= self.grasp_ticks:
                self.phase = 2
                self.step_in_phase = 0
            return SkillStatus.RUNNING

        elif self.phase == 2:
            # Standing back up with object clamped
            self.step_in_phase += 1
            if self.step_in_phase >= self.rise_ticks:
                self.phase = 0
                self.step_in_phase = 0
                return SkillStatus.SUCCESS
            return SkillStatus.RUNNING

        return SkillStatus.RUNNING

    def abort(self) -> None:
        self.phase = 0
        self.step_in_phase = 0


class MocapMotionActionNode(BehaviorNode):
    """
    Executes a retargeted human motion capture clip (e.g. Bandai Bow)
    streamed through the MocapPlayer at 50 Hz with stability-first projection.
    """

    def __init__(
        self,
        name: str,
        mocap_player: Any,
        blackboard: Optional[Blackboard] = None,
        joint_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        super().__init__(name, blackboard)
        self.player = mocap_player
        self.joint_callback = joint_callback
        self.started = False

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        stability = str(world_state.get("stability", "HIGH")).upper()
        if stability == "LOW":
            # Suppress mocap motion if balance is compromised
            return SkillStatus.RUNNING

        if not self.started:
            self.player.start()
            self.started = True

        done, joint_targets = self.player.step()
        if self.joint_callback is not None:
            self.joint_callback(joint_targets)

        if done:
            self.started = False
            return SkillStatus.SUCCESS

        return SkillStatus.RUNNING

    def abort(self) -> None:
        self.started = False


class VoiceResponseNode(BehaviorNode):
    """
    Triggers an authentic Microduck procedural vocalization (greet, inquire, peck, chirp, coo, wheee)
    when triggered by the Behavior Tree.
    """

    def __init__(
        self,
        name: str,
        recipe_name: str,
        blackboard: Optional[Blackboard] = None,
        sound_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(name, blackboard)
        self.recipe_name = recipe_name
        self.sound_callback = sound_callback
        self.emitted = False

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        if not self.emitted:
            if self.sound_callback is not None:
                self.sound_callback(self.recipe_name)
            self.emitted = True
            return SkillStatus.SUCCESS
        return SkillStatus.SUCCESS

    def abort(self) -> None:
        self.emitted = False


class VisualTrackApproachNode(BehaviorNode):
    """
    Visual servoing node: guides duck toward locked target using SORTTracker bearing and range.
    """

    def __init__(
        self,
        name: str,
        blackboard: Optional[Blackboard] = None,
        target_range_m: float = 0.18,
        velocity_callback: Optional[Callable[[float, float, float], None]] = None,
    ) -> None:
        super().__init__(name, blackboard)
        self.target_range_m = target_range_m
        self.velocity_callback = velocity_callback

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        locked = bool(world_state.get("target_locked", False))
        if not locked:
            if self.velocity_callback is not None:
                self.velocity_callback(0.0, 0.0, 0.05)
            return SkillStatus.RUNNING

        bearing = float(world_state.get("visual_bearing_rad", 0.0))
        dist = float(world_state.get("visual_range_m", 1.0))

        if (dist - self.target_range_m) <= 0.02 and abs(bearing) < 0.12:
            if self.velocity_callback is not None:
                self.velocity_callback(0.0, 0.0, 0.0)
            return SkillStatus.SUCCESS

        wz = float(np.clip(-0.75 * bearing, -0.15, 0.15))
        if abs(bearing) > 0.40:
            vx = 0.0
        else:
            vx = float(np.clip(0.30 * (dist - self.target_range_m), 0.0, 0.08))

        if self.velocity_callback is not None:
            self.velocity_callback(vx, 0.0, wz)

        return SkillStatus.RUNNING

    def abort(self) -> None:
        if self.velocity_callback is not None:
            self.velocity_callback(0.0, 0.0, 0.0)

