#!/usr/bin/env python3
"""
Pre-flight hardware check and diagnostic utility for physical Pollen Robotics Microduck.
Verifies connection to robotd and tofd, checks battery voltage, monitors joint temperatures,
validates ToF sensor streaming, and runs safe non-locomotive motion checks.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microduck_brain.pollen_bridge import PollenRobotdClient, PollenTofClient


def run_checks(
    robot_endpoint: str,
    tof_endpoint: str,
    dry_run: bool = False,
    run_actuation: bool = True,
) -> bool:
    print("=" * 60)
    print(" Pollen Robotics Microduck Hardware Pre-Flight Check")
    print("=" * 60)
    print(f"Target robotd endpoint: {robot_endpoint}")
    print(f"Target tofd endpoint:   {tof_endpoint}")
    print(f"Dry-run mode:           {dry_run}")
    print("-" * 60)

    if dry_run:
        print("[1/5] Connection check (mocked):       PASS (simulated)")
        print("[2/5] Battery voltage check (mocked):   PASS (7.42V - optimal)")
        print("[3/5] Safety state check (mocked):      PASS (upright, normal gains)")
        print("[4/5] ToF sensor check (mocked):        PASS (8x8 grid active, 1.25m min clearance)")
        if run_actuation:
            print("[5/5] Actuation smoke check (mocked):   PASS (beak, head, sound verified)")
        print("-" * 60)
        print("Pre-flight hardware check: ALL CHECKS PASSED")
        print("=" * 60)
        return True

    all_passed = True

    # 1. Connect to robotd
    print("[1/5] Connecting to robotd...", end=" ", flush=True)
    client = PollenRobotdClient(endpoint=robot_endpoint)
    if not client.connect(timeout=3.0):
        print("FAIL")
        print(f"  Error: Could not connect to {robot_endpoint}.")
        print("  Ensure robotd is running (`systemctl status robotd` or check network port).")
        return False
    print("PASS")

    try:
        mode = client.get_mode()
        print(f"  Daemon mode: {mode or 'unknown'}")

        model = client.get_model()
        if model and "result" in model:
            res = model["result"]
            print(f"  Asset: {res.get('asset', 'unknown')}, Joints: {len(res.get('joint_names', []))}")

        # 2. Battery & Telemetry check
        print("[2/5] Querying battery and safety status...", end=" ", flush=True)
        # Wait briefly for state notification
        telemetry: Dict[str, Any] = {}

        def on_state(state: Dict[str, Any]) -> None:
            nonlocal telemetry
            telemetry = state

        client.subscribe_state(on_state)
        time.sleep(0.5)

        if not telemetry and client.last_state:
            telemetry = client.last_state

        battery_volts = None
        if telemetry:
            bat = telemetry.get("battery")
            if isinstance(bat, dict):
                battery_volts = bat.get("volts")
            elif "battery_volts" in telemetry:
                battery_volts = float(telemetry["battery_volts"])

        if battery_volts is not None:
            if battery_volts < 6.5:
                print("FAIL")
                print(f"  CRITICAL: Battery bus voltage is {battery_volts:.2f}V (< 6.5V brownout limit).")
                print("  Do not operate locomotion. Charge or connect bench supply immediately.")
                all_passed = False
            elif battery_volts < 7.0:
                print("WARN")
                print(f"  WARNING: Battery bus voltage is {battery_volts:.2f}V (< 7.0V). Recharge soon.")
            else:
                print(f"PASS ({battery_volts:.2f}V)")
        else:
            print("WARN (battery telemetry not reported in initial frame)")

        # 3. Safety status
        print("[3/5] Inspecting safety flags...", end=" ", flush=True)
        safety = telemetry.get("safety", {}) if telemetry else {}
        is_fallen = safety.get("fallen", False)
        is_limp = safety.get("limp", False)
        gravity = safety.get("gravity", [0.0, 0.0, -1.0])

        if is_fallen:
            print("FAIL")
            print("  Robot reports fallen state. Place robot upright on a flat surface.")
            all_passed = False
        elif is_limp:
            print("WARN")
            print("  Servos are in limp mode (gains disabled).")
        else:
            print(f"PASS (upright, gravity: {gravity})")

        # 4. ToF sensor check
        print("[4/5] Connecting to tofd...", end=" ", flush=True)
        tof_client = PollenTofClient(endpoint=tof_endpoint)
        tof_connected = tof_client.connect(timeout=2.0)
        if tof_connected:
            time.sleep(0.4)
            tof_frame = tof_client.last_tof_frame
            if tof_frame and "distance_mm" in tof_frame:
                dist = tof_frame["distance_mm"]
                valid_dist = [d for d in dist if d > 0]
                min_d = min(valid_dist) if valid_dist else -1
                print(f"PASS ({len(dist)} zones, min clearance {min_d} mm)")
            else:
                print("WARN (connected, but no depth frame received yet)")
            tof_client.close()
        else:
            print(f"WARN (could not reach {tof_endpoint}, skipping ToF test)")

        # 5. Actuation smoke check (non-locomotive)
        if run_actuation:
            print("[5/5] Testing non-locomotive actuators (beak, head, sound)...", end=" ", flush=True)
            # Beak open slightly
            client.mouth(0.3)
            time.sleep(0.2)
            client.mouth(0.0)
            time.sleep(0.1)

            # Gentle head tilt
            client.head(neck_pitch=0.0, head_pitch=0.1, head_yaw=0.0, head_roll=0.0)
            time.sleep(0.25)
            client.head(neck_pitch=0.0, head_pitch=0.0, head_yaw=0.0, head_roll=0.0)
            time.sleep(0.1)

            # Sound
            client.play_sound("quack")
            client.stop()
            print("PASS")

    finally:
        client.stop()
        client.close()

    print("-" * 60)
    if all_passed:
        print("Pre-flight hardware check: ALL MANDATORY CHECKS PASSED")
    else:
        print("Pre-flight hardware check: ONE OR MORE CHECKS FAILED")
    print("=" * 60)
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Microduck hardware pre-flight check")
    parser.add_argument("--endpoint", "--socket", default="/run/robotd.sock", help="robotd socket or host:port")
    parser.add_argument("--tof-endpoint", "--tof-socket", default="/run/tofd/tof.sock", help="tofd socket or host:port")
    parser.add_argument("--dry-run", action="store_true", help="Simulate check for CI or offline testing")
    parser.add_argument("--no-actuate", action="store_true", help="Skip physical motion checks")
    args = parser.parse_args()

    success = run_checks(
        robot_endpoint=args.endpoint,
        tof_endpoint=args.tof_endpoint,
        dry_run=args.dry_run,
        run_actuation=not args.no_actuate,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
