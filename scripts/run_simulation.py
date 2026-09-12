"""
End-to-end simulation of the four-tier Microduck Robot Brain.
Demonstrates intent parsing, sensor debouncing, Behavior Tree execution
with ambient gestures and stability suppression, and locomotion command filtering.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Mapping, Any
from microduck_brain import (
    parse_voice_command_stateless,
    SchmittTrigger,
    TemporalDebouncer,
    SequenceNode,
    SelectorNode,
    AsymmetricParallelNode,
    SafeExpressionNode,
    AmbientBodyLanguageNode,
    FailureExpressionBranch,
    Blackboard,
    BehaviorNode,
    SkillStatus,
    MicroduckLocomotionEngine,
)


class SearchBallNode(BehaviorNode):
    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        if world_state.get("ball_visible", False):
            self.blackboard.reset_failure("search_ball")
            return SkillStatus.SUCCESS
        return SkillStatus.FAILURE


class ApproachTargetNode(BehaviorNode):
    def __init__(self, name: str, blackboard: Blackboard, target_dist: float = 0.25):
        super().__init__(name, blackboard)
        self.target_dist = target_dist

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        dist = float(world_state.get("distance", 99.0))
        if dist <= self.target_dist:
            return SkillStatus.SUCCESS
        if not world_state.get("ball_visible", False):
            return SkillStatus.FAILURE
        return SkillStatus.RUNNING


class PickupBallNode(BehaviorNode):
    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        dist = float(world_state.get("distance", 99.0))
        if dist <= 0.30 and world_state.get("ball_visible", False):
            return SkillStatus.SUCCESS
        return SkillStatus.FAILURE


def run_demo() -> None:
    print("=== Microduck Robot Brain End-to-End Simulation ===\n")

    # --------------------------------------------------------------------------
    # Tier 1: Stateless Intent Extraction
    # --------------------------------------------------------------------------
    user_speech = "Ducky, bring me the ball."
    intent = parse_voice_command_stateless(None, None, user_speech)
    print(f"[Tier 1] Parsed speech: '{user_speech}'")
    print(f"         Intent: action={intent.action}, target={intent.target}, urgency={intent.urgency}\n")

    # --------------------------------------------------------------------------
    # Tier 2: WorldState Sanitization Filters
    # --------------------------------------------------------------------------
    stability_filter = SchmittTrigger(low_threshold=0.35, high_threshold=0.65)
    ball_detector = TemporalDebouncer(window_size=5, required_ratio=0.7)

    # Raw noisy measurements from IMU and camera
    raw_imu_stream = [0.60, 0.68, 0.62, 0.66, 0.30]
    raw_vision_stream = [True, True, False, True, True]

    print("[Tier 2] Filtering noisy perception inputs:")
    clean_stability = "LOW"
    clean_ball_seen = False
    for i, (imu_val, vis_val) in enumerate(zip(raw_imu_stream, raw_vision_stream)):
        clean_stability = stability_filter.update(imu_val)
        clean_ball_seen = ball_detector.update(vis_val)
        print(f"  Sample {i+1}: raw IMU={imu_val:.2f} -> {clean_stability} | raw ball={vis_val} -> debounced={clean_ball_seen}")
    print()

    # --------------------------------------------------------------------------
    # Tier 3: Behavior Tree Assembly
    # --------------------------------------------------------------------------
    bb = Blackboard()
    engine = MicroduckLocomotionEngine()

    current_motion = {"vx": 0.0, "wz": 0.0, "head_yaw": 0.0, "spine_trim": 0.0}

    def on_expression_offset(yaw: float, spine: float):
        current_motion["head_yaw"] = yaw
        current_motion["spine_trim"] = spine

    def on_gesture(gesture_name: str):
        print(f"  [Actuator Bus] Executing body language: {gesture_name}")

    # Build Fetch Tree
    # 1. Search ball with fallback inquisitive head tilt or head shake
    # 2. Parallel approach: Walk to ball + ambient curious head scanning
    # 3. Pickup ball
    # 4. Celebration wiggle
    search_branch = SelectorNode("SearchOrExpress", bb, [
        SearchBallNode("SearchBall", bb),
        FailureExpressionBranch("ExpressSearchFailure", bb, "search_ball", gesture_callback=on_gesture),
    ])

    approach_parallel = AsymmetricParallelNode("WalkAndScan", bb, [
        ApproachTargetNode("ApproachTarget", bb, target_dist=0.25),
        AmbientBodyLanguageNode("AmbientScan", bb, frequency_hz=0.5, max_yaw_deg=10.0, offset_callback=on_expression_offset),
    ], primary_idx=0)

    fetch_tree = SequenceNode("FetchMission", bb, [
        search_branch,
        approach_parallel,
        PickupBallNode("PickupBall", bb),
        SafeExpressionNode("Celebrate", bb, gesture_id="HAPPY_WIGGLE", duration_ticks=3, gesture_callback=on_gesture),
    ])

    # --------------------------------------------------------------------------
    # Scenario A: Searching with ball not visible -> Inquisitive Head Tilt
    # --------------------------------------------------------------------------
    print("[Scenario A] Ball not visible, stable stance:")
    world_a = {"ball_visible": False, "stability": "HIGH", "roughness": "LOW", "distance": 1.5}
    status = fetch_tree.tick(world_a)
    print(f"  Tree Status: {status.name} (executed tilt expression)\n")

    # --------------------------------------------------------------------------
    # Scenario B: Slipped terrain during failure -> Expression suppressed safely
    # --------------------------------------------------------------------------
    print("[Scenario B] Ball not visible, terrain slipping (stability=LOW):")
    world_b = {"ball_visible": False, "stability": "LOW", "roughness": "HIGH", "distance": 1.5}
    status = fetch_tree.tick(world_b)
    print(f"  Tree Status: {status.name} (expression suppressed to preserve balance)\n")

    # --------------------------------------------------------------------------
    # Scenario C: Ball visible at 1.0 m -> Walking and ambient scanning
    # --------------------------------------------------------------------------
    print("[Scenario C] Ball visible, approaching target:")
    world_c = {"ball_visible": True, "stability": "HIGH", "roughness": "LOW", "distance": 1.0}
    status = fetch_tree.tick(world_c)
    print(f"  Tree Status: {status.name} | Head yaw trim: {current_motion['head_yaw']:.2f} deg")

    # Pass command to Tier 4 locomotion engine
    # (vx=0.4, vy=0.0, wz=0.0, roll=0.10 rad, pitch=0.0 rad)
    raw_cmd = (0.4, 0.0, 0.0, 0.10, 0.0)
    gravity = [0.0, 0.0, -1.0]
    lin_vel = [0.4, 0.0, 0.0]
    ang_vel = [0.0, 0.0, 0.0]
    joint_pos = list(engine.default_dof_pos)
    joint_vel = [0.0] * 12

    pd_targets = engine.step(
        raw_cmd=raw_cmd,
        projected_gravity=gravity,
        base_lin_vel=lin_vel,
        base_ang_vel=ang_vel,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )
    print(f"  [Tier 4] Computed 12-DOF PD targets (first 3 FL joints: {np.round(pd_targets[:3], 3)})\n")

    print("[Summary] All four tiers completed execution deterministically.")


if __name__ == "__main__":
    import numpy as np
    run_demo()
