#!/usr/bin/env python3
"""
Physical Microduck executive brain runner.
Integrates DuckBrain (Tier 1-3) directly with Pollen Robotics robotd and tofd daemons.
Runs continuous 20 Hz behavior tree ticks, streams sensor telemetry into WorldState,
and commands locomotion, head gaze, beak gestures, and episodic skills.
"""

import argparse
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Dict, Mapping, Optional

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microduck_brain.behavior_tree import (
    AmbientBodyLanguageNode,
    AsymmetricParallelNode,
    BehaviorNode,
    Blackboard,
    FailureExpressionBranch,
    ReactiveSelectorNode,
)
from microduck_brain.intent_parser import IntentParser
from microduck_brain.pollen_bridge import PollenRobotdClient, PollenTofClient
from microduck_brain.skill_latch import SkillStatus
from microduck_brain.vision.visual_servoing import VisualServoingController
from microduck_brain.world_state import WorldState


@dataclass
class _ServoTrackAdapter:
    target_locked: bool
    bearing_rad: float
    range_m: float


class _CallbackActionNode(BehaviorNode):
    """Behavior Tree leaf node invoking a stateful robot action callback."""

    def __init__(self, name: str, blackboard: Blackboard, fn: Any) -> None:
        super().__init__(name, blackboard)
        self._fn = fn

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        return self._fn(world_state)


class PhysicalMicroduckController:
    """Executive controller running DuckBrain against Pollen robotd and tofd."""

    def __init__(
        self,
        robot_endpoint: str = "/run/robotd.sock",
        tof_endpoint: str = "/run/tofd/tof.sock",
        mode: str = "walk",
        dry_run: bool = False,
    ) -> None:
        self.robot_endpoint = robot_endpoint
        self.tof_endpoint = tof_endpoint
        self.mode = mode
        self.dry_run = dry_run

        self.client = PollenRobotdClient(endpoint=robot_endpoint)
        self.tof_client = PollenTofClient(endpoint=tof_endpoint)

        self.intent_parser = IntentParser()
        self.world_state = WorldState()
        self.blackboard = Blackboard()

        self.active_intent = "WAIT"
        self.active_target: Optional[str] = None
        self.running = False
        self.head_trim = [0.0, 0.0]  # yaw_offset_rad, spine_trim_rad
        self.lock = threading.RLock()

        # Telemetry stats & watchdog
        self.ticks_count = 0
        self.last_telemetry_time = 0.0
        self.telemetry_timeout_s = 1.0
        self.last_active_node = "AmbientHeadScan"
        self.last_cmd = (0.0, 0.0, 0.0)
        self.event_log: deque[dict[str, Any]] = deque(maxlen=40)

        # Visual servoing + failure expression handling
        self.visual_servo = VisualServoingController()
        self._search_ticks = 0
        self.search_timeout_ticks = 200  # 10 s at 20 Hz before inquisitive head tilt/shake
        self._failure_expr = FailureExpressionBranch(
            "SearchFailureExpression",
            blackboard=self.blackboard,
            monitored_task="fetch_search",
            gesture_callback=self._on_expression_gesture,
        )

        # Real Tier-3 Behavior Tree wired per intent
        self._ambient_node = AmbientBodyLanguageNode(
            "AmbientHeadScan",
            blackboard=self.blackboard,
            frequency_hz=0.25,
            max_yaw_deg=8.0,
            offset_callback=self._on_ambient_offset,
        )
        self._trees: Dict[str, BehaviorNode] = self._build_behavior_trees()

    def _log_event(self, kind: str, message: str) -> None:
        self.event_log.append({
            "ts": round(time.time(), 2),
            "kind": kind,
            "message": message,
        })

    def _on_expression_gesture(self, gesture_id: str) -> None:
        self._log_event("expression", f"Triggered {gesture_id}")
        if not self.dry_run:
            if gesture_id == "HEAD_TILT":
                self.client.head(neck_pitch=0.0, head_pitch=0.05, head_yaw=0.0, head_roll=0.24)
            elif gesture_id == "HEAD_SHAKE":
                self.client.head(neck_pitch=0.0, head_pitch=0.0, head_yaw=0.28, head_roll=0.0)

    def _send_move(self, vx: float, vy: float, vyaw: float) -> None:
        self.last_cmd = (round(vx, 3), round(vy, 3), round(vyaw, 3))
        if not self.dry_run:
            self.client.move(vx=vx, vy=vy, vyaw=vyaw)

    def _send_stop(self) -> None:
        self.last_cmd = (0.0, 0.0, 0.0)
        if not self.dry_run:
            self.client.stop()

    def update_vision_observation(
        self,
        target_locked: bool,
        bearing_rad: float = 0.0,
        range_m: float = 1.0,
        track_id: Optional[int] = None,
    ) -> None:
        """Stream head-camera tracker detections directly into Tier-2 WorldState."""
        self.world_state.update_visual_tracking(
            target_locked=target_locked,
            bearing_rad=bearing_rad,
            range_m=range_m,
            track_id=track_id,
        )

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Return JSON-serializable state snapshot for live dashboard / monitoring."""
        with self.lock:
            ws = self.world_state.to_dict()
            return {
                "timestamp": time.time(),
                "mode": self.mode,
                "dry_run": self.dry_run,
                "running": self.running,
                "ticks": self.ticks_count,
                "active_intent": self.active_intent,
                "active_target": self.active_target,
                "active_node": self.last_active_node,
                "cmd": {
                    "vx": self.last_cmd[0],
                    "vy": self.last_cmd[1],
                    "vyaw": self.last_cmd[2],
                },
                "head_trim_rad": list(self.head_trim),
                "world_state": ws,
                "events": list(self.event_log)[-15:],
            }

    def _build_behavior_trees(self) -> Dict[str, BehaviorNode]:
        """Construct reactive Behavior Tree branches for every supported intent."""
        follow_primary = _CallbackActionNode("FollowApproach", self.blackboard, self._tick_follow)
        follow_tree = AsymmetricParallelNode(
            "FollowWithAmbient",
            self.blackboard,
            children=[follow_primary, self._ambient_node],
            primary_idx=0,
        )

        fetch_pickup = _CallbackActionNode("FetchPickup", self.blackboard, self._tick_fetch_pickup)
        fetch_approach = _CallbackActionNode("FetchApproach", self.blackboard, self._tick_fetch_approach)
        fetch_search = _CallbackActionNode("FetchSearch", self.blackboard, self._tick_fetch_search)
        fetch_tree = ReactiveSelectorNode(
            "FetchSelector",
            self.blackboard,
            children=[fetch_pickup, fetch_approach, fetch_search],
        )

        patrol_tree = AsymmetricParallelNode(
            "PatrolWithAmbient",
            self.blackboard,
            children=[
                _CallbackActionNode("PatrolMove", self.blackboard, self._tick_patrol),
                self._ambient_node,
            ],
            primary_idx=0,
        )

        goto_tree = _CallbackActionNode("GoToApproach", self.blackboard, self._tick_follow)
        wait_tree = self._ambient_node

        return {
            "FOLLOW": follow_tree,
            "FETCH": fetch_tree,
            "PATROL": patrol_tree,
            "GO_TO": goto_tree,
            "WAIT": wait_tree,
        }

    def _on_ambient_offset(self, yaw_deg: float, spine_deg: float) -> None:
        yaw_rad = math.radians(yaw_deg)
        spine_rad = math.radians(spine_deg)
        self.head_trim = [yaw_rad, spine_rad]
        if not self.dry_run:
            self.client.head(neck_pitch=0.0, head_pitch=0.05, head_yaw=yaw_rad, head_roll=0.0)

    def _tick_follow(self, ws_dict: Mapping[str, Any]) -> SkillStatus:
        self.last_active_node = "FollowApproach"
        if ws_dict.get("obstacle_close", False):
            self._send_stop()
            return SkillStatus.RUNNING
        self._send_move(vx=0.10, vy=0.0, vyaw=0.0)
        return SkillStatus.RUNNING

    def _tick_patrol(self, ws_dict: Mapping[str, Any]) -> SkillStatus:
        self.last_active_node = "PatrolMove"
        if ws_dict.get("obstacle_close", False):
            self._send_move(vx=0.0, vy=0.0, vyaw=0.30)
            return SkillStatus.RUNNING
        self._send_move(vx=0.08, vy=0.0, vyaw=0.0)
        return SkillStatus.RUNNING

    def _complete_fetch_pickup(self) -> SkillStatus:
        self.last_active_node = "FetchPickup"
        self._search_ticks = 0
        self.blackboard.reset_failure("fetch_search")
        self._send_stop()
        if not self.dry_run:
            self.client.do_skill("ground_pick")
            self.client.play_sound("quack")
        self._log_event("skill", "Completed fetch -> ground_pick + quack")
        with self.lock:
            self.active_intent = "WAIT"
            self.active_target = None
        return SkillStatus.SUCCESS

    def _tick_fetch_pickup(self, ws_dict: Mapping[str, Any]) -> SkillStatus:
        if not (ws_dict.get("ball_visible", False) and ws_dict.get("obstacle_close", False)):
            return SkillStatus.FAILURE
        return self._complete_fetch_pickup()

    def _tick_fetch_approach(self, ws_dict: Mapping[str, Any]) -> SkillStatus:
        if not ws_dict.get("ball_visible", False):
            return SkillStatus.FAILURE
        self.last_active_node = "FetchApproach"
        self._search_ticks = 0
        # Use closed-loop visual servoing when bearing/range track lock is active
        if ws_dict.get("target_locked", False):
            track = _ServoTrackAdapter(
                target_locked=True,
                bearing_rad=float(ws_dict.get("visual_bearing_rad", 0.0)),
                range_m=float(ws_dict.get("visual_range_m", 1.0)),
            )
            vx, wz, reached = self.visual_servo.compute_commands(track)
            if reached:
                return self._complete_fetch_pickup()
            self._send_move(vx=vx, vy=0.0, vyaw=wz)
            return SkillStatus.RUNNING

        self._send_move(vx=0.12, vy=0.0, vyaw=0.0)
        return SkillStatus.RUNNING

    def _tick_fetch_search(self, ws_dict: Mapping[str, Any]) -> SkillStatus:
        self.last_active_node = "FetchSearch"
        self._search_ticks += 1
        if self._search_ticks >= self.search_timeout_ticks:
            self._search_ticks = 0
            self.last_active_node = "SearchFailureExpression"
            self._send_stop()
            self._failure_expr.tick(ws_dict)
            return SkillStatus.RUNNING
        self._send_move(vx=0.0, vy=0.0, vyaw=0.25)
        if not self.dry_run:
            self.client.head(neck_pitch=0.0, head_pitch=0.15, head_yaw=0.0, head_roll=0.0)
        return SkillStatus.RUNNING

    def _abort_all_trees(self) -> None:
        self._search_ticks = 0
        self._failure_expr.abort()
        for tree in self._trees.values():
            tree.abort()

    def start(self) -> bool:
        if self.dry_run:
            print(f"[DuckBrain] Running in dry-run mode (simulating physical {self.robot_endpoint}).")
            self.running = True
            return True

        print(f"[DuckBrain] Connecting to robotd at {self.robot_endpoint}...")
        if not self.client.connect(timeout=3.0):
            print(f"[DuckBrain] ERROR: Failed to connect to robotd at {self.robot_endpoint}.")
            return False

        # Verify or set mode
        current_mode = self.client.get_mode()
        print(f"[DuckBrain] Active robotd drive mode: {current_mode}")
        if self.mode and current_mode != self.mode:
            print(f"[DuckBrain] Setting mode to '{self.mode}'...")
            self.client.set_mode(self.mode)

        # Subscribe to robot.state telemetry
        self.client.subscribe_state(self._on_robot_state)

        # Connect to tofd
        print(f"[DuckBrain] Connecting to tofd at {self.tof_endpoint}...")
        if self.tof_client.connect(timeout=2.0):
            self.tof_client.tof_callback = self._on_tof_frame
            print("[DuckBrain] ToF 8x8 depth matrix subscription active.")
        else:
            print(f"[DuckBrain] Warning: tofd not reachable at {self.tof_endpoint}, running without ToF.")

        self.running = True
        return True

    def stop(self) -> None:
        self.running = False
        self._abort_all_trees()
        print("\n[DuckBrain] Stopping physical duck and zeroing velocity...")
        if not self.dry_run:
            try:
                self.client.stop()
                self.client.head(0.0, 0.0, 0.0, 0.0)
                self.client.mouth(0.0)
            except Exception:
                pass
            self.client.close()
            self.tof_client.close()
        print("[DuckBrain] Shutdown complete.")

    def set_intent(self, user_command: str) -> str:
        """Parses natural language intent and updates active plan."""
        cleaned = user_command.strip().lower()
        if cleaned in ("wait", "idle"):
            with self.lock:
                self._abort_all_trees()
                self.active_intent = "WAIT"
                self.active_target = None
                self.last_active_node = "AmbientHeadScan"
                self._log_event("intent", "WAIT")
            print(f"[DuckBrain] Intent: WAIT (raw: '{user_command}')")
            return "WAIT"

        parsed = self.intent_parser.parse(user_command)
        intent = str(parsed.get("intent", "WAIT"))
        target = parsed.get("target")
        print(f"[DuckBrain] Intent: {intent} target={target} (raw: '{user_command}')")

        with self.lock:
            self._abort_all_trees()
            self.active_intent = intent
            self.active_target = target
            self._log_event("intent", f"{intent}" + (f" ({target})" if target else ""))

            # Instant hard-stop deterministic bypass
            if intent == "STOP":
                self.last_active_node = "EmergencyStop"
                self._send_stop()
                print("[DuckBrain] HARD EMERGENCY STOP DISPATCHED")
            elif intent == "SIT":
                self.last_active_node = "Skill:sit_toggle"
                self._send_stop()
                if not self.dry_run:
                    self.client.do_skill("sit_toggle")
                self.active_intent = "WAIT"
            elif intent == "QUACK":
                self.last_active_node = "Voice:quack"
                if not self.dry_run:
                    self.client.play_sound(str(target or "quack"))
                self.active_intent = "WAIT"
            elif intent == "SKILL":
                skill_name = str(target or "bow")
                self.last_active_node = f"Skill:{skill_name}"
                self._send_stop()
                if not self.dry_run:
                    self.client.do_skill(skill_name)
                print(f"[DuckBrain] Dispatched episodic skill '{skill_name}'")
                self.active_intent = "WAIT"

        return intent

    def _on_robot_state(self, state: Dict[str, Any]) -> None:
        self.last_telemetry_time = time.monotonic()
        self.world_state.update_from_robot_state(state)

    def _on_tof_frame(self, frame: Dict[str, Any]) -> None:
        self.world_state.update_from_tof_frame(frame)

    def tick_behavior_tree(self) -> None:
        """Executes one 20 Hz behavior tree iteration."""
        self.ticks_count += 1
        ws_dict = self.world_state.to_dict()

        with self.lock:
            # 0. Telemetry staleness watchdog (active once telemetry has started)
            if (
                not self.dry_run
                and self.last_telemetry_time > 0.0
                and (time.monotonic() - self.last_telemetry_time) > self.telemetry_timeout_s
            ):
                self._abort_all_trees()
                self.last_active_node = "TelemetryWatchdog"
                if self.ticks_count % 40 == 0:
                    print("[DuckBrain] TELEMETRY WATCHDOG: robotd telemetry stale. Locomotion halted.")
                self._send_stop()
                return

            # 1. Hardware brownout safety guardrail
            if ws_dict.get("is_brownout_risk", False):
                self._abort_all_trees()
                self.last_active_node = "BrownoutGuard"
                if self.ticks_count % 40 == 0:
                    print(f"[DuckBrain] BROWNOUT GUARD: Voltage {ws_dict['battery_volts']:.2f}V <= 6.5V. Locomotion locked.")
                self._send_stop()
                return

            # 2. Fall detection guardrail
            if ws_dict.get("is_fallen", False):
                self._abort_all_trees()
                self.last_active_node = "FallGuard"
                if self.ticks_count % 40 == 0:
                    print("[DuckBrain] FALL GUARD: Robot reports fallen state. Locomotion suspended.")
                self._send_stop()
                return

            # 3. Intent execution through active Behavior Tree
            intent = self.active_intent
            if intent == "STOP":
                self.last_active_node = "EmergencyStop"
                self._send_stop()
                return

            if intent == "WAIT":
                self.last_active_node = "AmbientHeadScan"
            tree = self._trees.get(intent, self._trees["WAIT"])
            tree.tick(ws_dict)

    def run_loop(self) -> None:
        """Main 20 Hz executive control loop."""
        period = 0.05  # 20 Hz
        while self.running:
            start_t = time.time()
            self.tick_behavior_tree()
            elapsed = time.time() - start_t
            sleep_t = max(0.001, period - elapsed)
            time.sleep(sleep_t)


def interactive_repl(controller: PhysicalMicroduckController) -> None:
    """Console REPL for issuing commands to the physical duck."""
    print("\n" + "=" * 60)
    print(" DuckBrain Physical Microduck Interactive Console")
    print(" Commands: come, follow, fetch ball, bow, kick, sit, quack,")
    print("           patrol, wait, stop, status, exit")
    print("=" * 60 + "\n")

    while controller.running:
        try:
            line = input("ducky> ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                controller.stop()
                break
            elif line.lower() == "status":
                ws = controller.world_state.to_dict()
                print(f"  [Status] Intent: {controller.active_intent}")
                print(f"  [Status] Battery: {ws['battery_volts']:.2f}V (sag risk: {ws['is_brownout_risk']})")
                print(f"  [Status] Stability: {ws['stability']}, Roughness: {ws['roughness']}")
                print(f"  [Status] Fallen: {ws['is_fallen']}, Obstacle close: {ws['obstacle_close']}")
                print(f"  [Status] Forward clearance: {ws['forward_clearance_m']:.2f} m")
            else:
                controller.set_intent(line)
        except (EOFError, KeyboardInterrupt):
            controller.stop()
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="DuckBrain Physical Microduck Runner")
    parser.add_argument("--endpoint", "--socket", default="/run/robotd.sock", help="robotd socket path or host:port")
    parser.add_argument("--tof-endpoint", "--tof-socket", default="/run/tofd/tof.sock", help="tofd socket or host:port")
    parser.add_argument("--mode", default="walk", choices=["walk", "roller"], help="Drive mode")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive text REPL")
    parser.add_argument("--dry-run", action="store_true", help="Simulate physical connection for verification")
    parser.add_argument("--dashboard", action="store_true", help="Launch live web telemetry & command cockpit")
    parser.add_argument("--dashboard-host", default="127.0.0.1", help="Dashboard bind host (default: 127.0.0.1)")
    parser.add_argument("--dashboard-port", type=int, default=8765, help="Dashboard HTTP port (default: 8765)")
    args = parser.parse_args()

    controller = PhysicalMicroduckController(
        robot_endpoint=args.endpoint,
        tof_endpoint=args.tof_endpoint,
        mode=args.mode,
        dry_run=args.dry_run,
    )

    dashboard = None

    def shutdown_all() -> None:
        if dashboard is not None:
            dashboard.stop()
        controller.stop()

    def handle_signal(sig, frame):
        shutdown_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if not controller.start():
        sys.exit(1)

    if args.dashboard:
        from microduck_brain.dashboard import DashboardServer
        dashboard = DashboardServer(
            snapshot_fn=controller.get_telemetry_snapshot,
            command_fn=controller.set_intent,
            host=args.dashboard_host,
            port=args.dashboard_port,
        )
        url = dashboard.start()
        print(f"[DuckBrain] Live telemetry cockpit running at {url}")

    try:
        # Spawn loop in background thread if interactive REPL is requested
        if args.interactive:
            loop_thread = threading.Thread(target=controller.run_loop, daemon=True)
            loop_thread.start()
            interactive_repl(controller)
        else:
            print("[DuckBrain] Running 20 Hz executive brain loop. Press Ctrl+C to stop.")
            controller.run_loop()
    finally:
        if dashboard is not None:
            dashboard.stop()


if __name__ == "__main__":
    main()
