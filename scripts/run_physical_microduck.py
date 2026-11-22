#!/usr/bin/env python3
"""
Physical Microduck executive brain runner.
Integrates DuckBrain (Tier 1-3) directly with Pollen Robotics robotd and tofd daemons.
Runs continuous 20 Hz behavior tree ticks, streams sensor telemetry into WorldState,
and commands locomotion, head gaze, beak gestures, and episodic skills.
"""

import argparse
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
    SafeExpressionNode,
    SelectorNode,
    SequenceNode,
)
from microduck_brain.intent_parser import IntentParser
from microduck_brain.pollen_bridge import PollenRobotdClient, PollenTofClient
from microduck_brain.skill_latch import SkillStatus
from microduck_brain.world_state import WorldState


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

        # Real Tier-3 Behavior Tree wired per intent
        self._ambient_node = AmbientBodyLanguageNode(
            "AmbientHeadScan",
            blackboard=self.blackboard,
            frequency_hz=0.25,
            max_yaw_deg=8.0,
            offset_callback=self._on_ambient_offset,
        )
        self._trees: Dict[str, BehaviorNode] = self._build_behavior_trees()

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
        if ws_dict.get("obstacle_close", False):
            if not self.dry_run:
                self.client.stop()
            return SkillStatus.RUNNING
        if not self.dry_run:
            self.client.move(vx=0.10, vy=0.0, vyaw=0.0)
        return SkillStatus.RUNNING

    def _tick_patrol(self, ws_dict: Mapping[str, Any]) -> SkillStatus:
        if ws_dict.get("obstacle_close", False):
            if not self.dry_run:
                self.client.move(vx=0.0, vy=0.0, vyaw=0.30)
            return SkillStatus.RUNNING
        if not self.dry_run:
            self.client.move(vx=0.08, vy=0.0, vyaw=0.0)
        return SkillStatus.RUNNING

    def _tick_fetch_pickup(self, ws_dict: Mapping[str, Any]) -> SkillStatus:
        if not (ws_dict.get("ball_visible", False) and ws_dict.get("obstacle_close", False)):
            return SkillStatus.FAILURE
        if not self.dry_run:
            self.client.stop()
            self.client.do_skill("ground_pick")
            self.client.play_sound("quack")
        with self.lock:
            self.active_intent = "WAIT"
            self.active_target = None
        return SkillStatus.SUCCESS

    def _tick_fetch_approach(self, ws_dict: Mapping[str, Any]) -> SkillStatus:
        if not ws_dict.get("ball_visible", False):
            return SkillStatus.FAILURE
        if not self.dry_run:
            self.client.move(vx=0.12, vy=0.0, vyaw=0.0)
        return SkillStatus.RUNNING

    def _tick_fetch_search(self, ws_dict: Mapping[str, Any]) -> SkillStatus:
        _ = ws_dict
        if not self.dry_run:
            self.client.move(vx=0.0, vy=0.0, vyaw=0.25)
            self.client.head(neck_pitch=0.0, head_pitch=0.15, head_yaw=0.0, head_roll=0.0)
        return SkillStatus.RUNNING

    def _abort_all_trees(self) -> None:
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

            # Instant hard-stop deterministic bypass
            if intent == "STOP":
                if not self.dry_run:
                    self.client.stop()
                print("[DuckBrain] HARD EMERGENCY STOP DISPATCHED")
            elif intent == "SIT":
                if not self.dry_run:
                    self.client.stop()
                    self.client.do_skill("sit_toggle")
                self.active_intent = "WAIT"
            elif intent == "QUACK":
                if not self.dry_run:
                    self.client.play_sound(str(target or "quack"))
                self.active_intent = "WAIT"
            elif intent == "SKILL":
                skill_name = str(target or "bow")
                if not self.dry_run:
                    self.client.stop()
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
                if self.ticks_count % 40 == 0:
                    print("[DuckBrain] TELEMETRY WATCHDOG: robotd telemetry stale. Locomotion halted.")
                self.client.stop()
                return

            # 1. Hardware brownout safety guardrail
            if ws_dict.get("is_brownout_risk", False):
                self._abort_all_trees()
                if self.ticks_count % 40 == 0:
                    print(f"[DuckBrain] BROWNOUT GUARD: Voltage {ws_dict['battery_volts']:.2f}V <= 6.5V. Locomotion locked.")
                if not self.dry_run:
                    self.client.stop()
                return

            # 2. Fall detection guardrail
            if ws_dict.get("is_fallen", False):
                self._abort_all_trees()
                if self.ticks_count % 40 == 0:
                    print("[DuckBrain] FALL GUARD: Robot reports fallen state. Locomotion suspended.")
                if not self.dry_run:
                    self.client.stop()
                return

            # 3. Intent execution through active Behavior Tree
            intent = self.active_intent
            if intent == "STOP":
                if not self.dry_run:
                    self.client.stop()
                return

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
    args = parser.parse_args()

    controller = PhysicalMicroduckController(
        robot_endpoint=args.endpoint,
        tof_endpoint=args.tof_endpoint,
        mode=args.mode,
        dry_run=args.dry_run,
    )

    def handle_signal(sig, frame):
        controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if not controller.start():
        sys.exit(1)

    # Spawn loop in background thread if interactive REPL is requested
    if args.interactive:
        loop_thread = threading.Thread(target=controller.run_loop, daemon=True)
        loop_thread.start()
        interactive_repl(controller)
    else:
        print("[DuckBrain] Running 20 Hz executive brain loop. Press Ctrl+C to stop.")
        controller.run_loop()


if __name__ == "__main__":
    main()
