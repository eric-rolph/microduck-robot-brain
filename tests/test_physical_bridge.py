"""
Unit tests for Pollen Robotics physical hardware bridge and WorldState ingestion.
"""

from unittest.mock import MagicMock
import pytest
from microduck_brain.pollen_bridge import PollenRobotdClient, parse_endpoint
from microduck_brain.world_state import WorldState
from scripts.run_physical_microduck import PhysicalMicroduckController


def test_parse_endpoint():
    # Unix socket paths
    family, addr = parse_endpoint("/run/robotd.sock")
    assert family == "unix"
    assert addr is None

    # TCP host and port
    family, addr = parse_endpoint("192.168.1.100:8088")
    assert family == "tcp"
    assert addr == ("192.168.1.100", 8088)

    family, addr = parse_endpoint("tcp://microduck.local:8088")
    assert family == "tcp"
    assert addr == ("microduck.local", 8088)

    # Windows file path should not be parsed as TCP
    family, addr = parse_endpoint("C:\\Users\\test\\robotd.sock")
    assert family == "unix"


def test_world_state_telemetry_orientation_and_safety():
    ws = WorldState()

    # Normal upright posture: gravity is [0, 0, -1]
    ws.update_from_robot_state({
        "safety": {
            "fallen": False,
            "limp": False,
            "gravity": [0.0, 0.0, -1.0],
        },
        "battery": {
            "volts": 7.4,
            "percent": 80.0,
        },
    })
    d = ws.to_dict()
    assert d["is_fallen"] is False
    assert d["is_limp"] is False
    assert d["stability"] == "HIGH"
    assert d["battery_volts"] == 7.4
    assert d["is_brownout_risk"] is False

    # Fallen state
    ws.update_from_robot_state({
        "safety": {
            "fallen": True,
            "limp": True,
            "gravity": [0.0, -1.0, 0.0],
        },
    })
    d = ws.to_dict()
    assert d["is_fallen"] is True
    assert d["is_limp"] is True


def test_world_state_battery_brownout_hysteresis():
    ws = WorldState()

    # Initial normal voltage
    ws.update_battery_voltage(7.4)
    assert ws.is_brownout_risk is False

    # Brownout condition (<= 6.5V)
    ws.update_battery_voltage(6.4)
    assert ws.is_brownout_risk is True

    # Voltage sags up slightly to 6.6V (below 6.8V upper threshold)
    ws.update_battery_voltage(6.6)
    assert ws.is_brownout_risk is True  # Hysteresis keeps lockout active

    # Full recovery to 7.2V
    ws.update_battery_voltage(7.2)
    assert ws.is_brownout_risk is False


def test_world_state_tof_depth_processing():
    ws = WorldState()

    # 8x8 matrix = 64 zones with 1500 mm depth
    tof_frame_clear = {
        "rows": 8,
        "cols": 8,
        "distance_mm": [1500] * 64,
    }
    clearance = ws.update_from_tof_frame(tof_frame_clear)
    assert clearance == 1.5

    # Center zone obstacle at 200 mm (0.2m)
    depths = [1500] * 64
    for r in range(2, 6):
        for c in range(2, 6):
            depths[r * 8 + c] = 200

    tof_frame_obstacle = {
        "rows": 8,
        "cols": 8,
        "distance_mm": depths,
    }
    for _ in range(5):
        ws.update_from_tof_frame(tof_frame_obstacle)

    assert ws.forward_clearance_m == 0.2
    assert ws.obstacle_close is True


def test_physical_controller_safety_locks():
    ctrl = PhysicalMicroduckController(dry_run=True)
    ctrl.start()
    ctrl.set_intent("come")

    # Set brownout condition
    ctrl.world_state.update_battery_voltage(6.2)
    assert ctrl.world_state.is_brownout_risk is True

    # Tick should not crash and should refuse locomotion
    ctrl.tick_behavior_tree()

    # Set fallen condition
    ctrl.world_state.update_battery_voltage(7.4)
    ctrl.world_state.is_fallen = True
    ctrl.tick_behavior_tree()

    ctrl.stop()
