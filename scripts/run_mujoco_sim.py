"""End-to-end simulation runner coupling DuckBrain with MuJoCo physics.

Drives the 50 Hz sim-to-real physics loop, translates Behavior Tree intent
into standardized 13-D motion commands, and evaluates BAM M6 actuator dynamics.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from microduck_brain.behavior_tree import (
    BehaviorNode,
    Blackboard,
    SequenceNode,
)
from microduck_brain.skill_latch import SkillStatus
from microduck_brain.sim.env import MicroduckMuJoCoEnv
from microduck_brain.world_state import WorldState


class SearchActionNode(BehaviorNode):
    """Behavior Tree node commanding in-place yaw rotation to locate target."""

    def __init__(self, name: str, blackboard: Blackboard, search_duration: float = 1.0) -> None:
        super().__init__(name, blackboard)
        self.search_duration = search_duration
        self.elapsed = 0.0

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        self.elapsed += 0.02
        if self.blackboard.data.get("ball_visible", False):
            self.blackboard.data["twist_cmd"] = (0.0, 0.0, 0.0)
            return SkillStatus.SUCCESS

        # Turn in place looking for the object
        self.blackboard.data["twist_cmd"] = (0.0, 0.0, 0.6)
        # Ambient curiosity head scan
        yaw_trim = 0.3 * math.sin(self.elapsed * 4.0)
        self.blackboard.data["head_cmd"] = (0.2, 0.1, yaw_trim, 0.0)

        if self.elapsed >= self.search_duration:
            # Simulate locating object
            self.blackboard.data["ball_visible"] = True
            return SkillStatus.SUCCESS
        return SkillStatus.RUNNING


class ApproachActionNode(BehaviorNode):
    """Behavior Tree node walking forward until object is within grasp range."""

    def __init__(self, name: str, blackboard: Blackboard, approach_duration: float = 1.5) -> None:
        super().__init__(name, blackboard)
        self.approach_duration = approach_duration
        self.elapsed = 0.0

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        self.elapsed += 0.02
        if self.elapsed >= self.approach_duration:
            self.blackboard.data["twist_cmd"] = (0.0, 0.0, 0.0)
            self.blackboard.data["ball_close"] = True
            return SkillStatus.SUCCESS

        # Forward walking command
        self.blackboard.data["twist_cmd"] = (0.2, 0.0, 0.0)
        # Head tilted downward toward ground object
        self.blackboard.data["head_cmd"] = (0.35, 0.25, 0.0, 0.0)
        return SkillStatus.RUNNING


class PickupActionNode(BehaviorNode):
    """Behavior Tree node commanding ground pick posture."""

    def __init__(self, name: str, blackboard: Blackboard, duration: float = 0.8) -> None:
        super().__init__(name, blackboard)
        self.duration = duration
        self.elapsed = 0.0

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        self.elapsed += 0.02
        self.blackboard.data["twist_cmd"] = (0.0, 0.0, 0.0)
        # Lower trunk and dip beak
        self.blackboard.data["head_cmd"] = (0.5, 0.4, 0.0, 0.0)
        self.blackboard.data["body_cmd"] = (0.0, 0.0, -0.02, 0.0, 0.15, 0.0)

        if self.elapsed >= self.duration:
            self.blackboard.data["ball_held"] = True
            self.blackboard.data["body_cmd"] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return SkillStatus.SUCCESS
        return SkillStatus.RUNNING


def build_fetch_tree(blackboard: Blackboard) -> SequenceNode:
    """Build Behavior Tree for 'FETCH_BALL' goal."""
    return SequenceNode(
        "FetchSequence",
        blackboard,
        [
            SearchActionNode("Search", blackboard),
            ApproachActionNode("Approach", blackboard),
            PickupActionNode("Pickup", blackboard),
        ],
    )


def run_simulation(
    steps: int = 150,
    onnx_path: str | Path | None = None,
    headless: bool = True,
    render: bool = False,
    dashboard: bool = False,
    dashboard_host: str = "127.0.0.1",
    dashboard_port: int = 8765,
) -> dict[str, float]:
    """Run DuckBrain behavior loop inside MuJoCo physics."""
    print(f"Initializing Microduck MuJoCo simulation (steps={steps})...")
    env = MicroduckMuJoCoEnv(use_backlash=True)
    obs = env.reset()

    # Load ONNX policy session if provided
    ort_session = None
    if onnx_path and Path(onnx_path).exists() and ort is not None:
        print(f"Loading ONNX policy from {onnx_path}...")
        ort_session = ort.InferenceSession(str(onnx_path))
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
    else:
        print("Running in stance hold mode.")

    # Initialize Blackboard and Behavior Tree
    bb = Blackboard()
    bb.data["twist_cmd"] = (0.0, 0.0, 0.0)
    bb.data["head_cmd"] = (0.0, 0.0, 0.0, 0.0)
    bb.data["body_cmd"] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    tree = build_fetch_tree(bb)
    world_state = WorldState()

    step_latencies: list[float] = []
    battery_voltages: list[float] = []
    trunk_heights: list[float] = []
    tree_statuses: list[str] = []

    dash_state: dict[str, Any] = {
        "timestamp": time.time(),
        "mode": "mujoco-sim",
        "dry_run": True,
        "running": True,
        "ticks": 0,
        "active_intent": "FETCH",
        "active_target": "ball",
        "active_node": "Search",
        "cmd": {"vx": 0.0, "vy": 0.0, "vyaw": 0.0},
        "head_trim_rad": [0.0, 0.0],
        "world_state": world_state.to_dict(),
        "events": [{"ts": round(time.time(), 2), "kind": "sim", "message": "MuJoCo 50 Hz session started"}],
    }
    dash_server = None
    if dashboard:
        from microduck_brain.dashboard import DashboardServer
        dash_server = DashboardServer(
            snapshot_fn=lambda: dict(dash_state),
            host=dashboard_host,
            port=dashboard_port,
        )
        print(f"[DuckBrain] Live simulation telemetry cockpit running at {dash_server.start()}")

    # Optional viewer
    viewer = None
    if render and not headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    print("Executing Behavior Tree driven simulation loop at 50 Hz...")
    try:
        for step in range(steps):
            t_start = time.perf_counter()

            # 1. Update world state from MuJoCo sensor data
            proj_grav = obs[3:6]
            world_state.update_orientation(roll=float(proj_grav[1]), pitch=float(proj_grav[0]))
            current_ws = world_state.to_dict()

            # 2. Tick Behavior Tree at 50 Hz
            bt_status = tree.tick(current_ws)
            tree_statuses.append(bt_status.name)

            # 3. Extract motion commands from Blackboard
            vx, vy, wz = bb.data.get("twist_cmd", (0.0, 0.0, 0.0))
            head_cmd = np.array(bb.data.get("head_cmd", (0.0, 0.0, 0.0, 0.0)), dtype=np.float32)
            body_cmd = np.array(bb.data.get("body_cmd", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), dtype=np.float32)

            env.set_command(
                lin_vel_x=vx,
                lin_vel_y=vy,
                ang_vel_z=wz,
                head_pose=head_cmd,
                body_pose=body_cmd,
            )

            # 4. Policy inference
            if ort_session is not None:
                batch_obs = obs.reshape(1, -1).astype(np.float32)
                raw_action = ort_session.run([output_name], {input_name: batch_obs})[0]
                action = raw_action.squeeze(0).astype(np.float32)
            else:
                action = np.zeros(14, dtype=np.float32)

            # 5. Step physics (10 sub-steps through BAM M6 and backlash)
            obs, reward, terminated, truncated, info = env.step(action)

            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            step_latencies.append(t_elapsed_ms)
            battery_voltages.append(info["battery_voltage"])
            trunk_heights.append(info["trunk_height"])
            world_state.update_battery_voltage(info["battery_voltage"])
            active_child = tree.children[min(tree.current_idx, len(tree.children) - 1)].name
            dash_state.update({
                "timestamp": time.time(),
                "ticks": step + 1,
                "active_node": active_child,
                "cmd": {"vx": float(vx), "vy": float(vy), "vyaw": float(wz)},
                "world_state": world_state.to_dict(),
            })

            # 6. Update viewer if active
            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(max(0.0, 0.02 - (t_elapsed_ms / 1000.0)))
            elif dash_server is not None:
                time.sleep(max(0.0, 0.02 - (t_elapsed_ms / 1000.0)))

            if terminated:
                print(f"Simulation terminated: robot fell at step {step}")
                break
    finally:
        if dash_server is not None:
            dash_server.stop()

    summary = {
        "completed_steps": len(step_latencies),
        "mean_latency_ms": float(np.mean(step_latencies)),
        "max_latency_ms": float(np.max(step_latencies)),
        "min_battery_voltage": float(np.min(battery_voltages)),
        "final_trunk_height": float(trunk_heights[-1]) if trunk_heights else 0.0,
        "tree_final_status": tree_statuses[-1] if tree_statuses else "NONE",
    }

    print("\n--- Simulation Summary ---")
    print(f"Steps executed:     {summary['completed_steps']}")
    print(f"Mean cycle time:    {summary['mean_latency_ms']:.2f} ms")
    print(f"Max cycle time:     {summary['max_latency_ms']:.2f} ms")
    print(f"Min battery sag:    {summary['min_battery_voltage']:.2f} V")
    print(f"Final trunk height: {summary['final_trunk_height']:.3f} m")
    print(f"BT completion:      {summary['tree_final_status']}")
    print("--------------------------\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Microduck MuJoCo simulation runner")
    parser.add_argument("--steps", type=int, default=150, help="Number of simulation steps")
    parser.add_argument("--onnx", type=str, default="models/microduck_walk.onnx", help="Path to ONNX policy")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without rendering window")
    parser.add_argument("--render", action="store_true", help="Launch MuJoCo interactive viewer")
    parser.add_argument("--dashboard", action="store_true", help="Launch live web telemetry cockpit")
    parser.add_argument("--dashboard-host", default="127.0.0.1", help="Dashboard bind host")
    parser.add_argument("--dashboard-port", type=int, default=8765, help="Dashboard HTTP port")
    args = parser.parse_args()

    run_simulation(
        steps=args.steps,
        onnx_path=args.onnx,
        headless=not args.render,
        render=args.render,
        dashboard=args.dashboard,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
    )


if __name__ == "__main__":
    main()
