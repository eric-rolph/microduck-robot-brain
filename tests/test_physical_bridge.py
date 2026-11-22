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


def test_physical_controller_intent_dispatch_and_behavior_trees():
    ctrl = PhysicalMicroduckController(dry_run=False)
    ctrl.client = MagicMock()
    ctrl.tof_client = MagicMock()
    ctrl.running = True
    ctrl.world_state.update_battery_voltage(7.6)

    # 1. 'quack' dispatches play_sound("quack"), NOT emergency stop
    assert ctrl.set_intent("quack") == "QUACK"
    ctrl.client.play_sound.assert_called_once_with("quack")
    ctrl.client.stop.assert_not_called()

    # 2. 'bow' and 'kick' dispatch stop() + do_skill(...)
    ctrl.client.reset_mock()
    assert ctrl.set_intent("bow") == "SKILL"
    ctrl.client.stop.assert_called_once()
    ctrl.client.do_skill.assert_called_once_with("bow")

    ctrl.client.reset_mock()
    assert ctrl.set_intent("kick") == "SKILL"
    ctrl.client.stop.assert_called_once()
    ctrl.client.do_skill.assert_called_once_with("kick")

    # 3. 'come' -> FOLLOW BT moves forward when clear, stops when obstacle close
    ctrl.client.reset_mock()
    assert ctrl.set_intent("come") == "FOLLOW"
    ctrl.world_state.obstacle_close = False
    ctrl.tick_behavior_tree()
    ctrl.client.move.assert_called_with(vx=0.10, vy=0.0, vyaw=0.0)

    ctrl.client.reset_mock()
    ctrl.world_state.obstacle_close = True
    ctrl.tick_behavior_tree()
    ctrl.client.stop.assert_called_once()

    # 4. 'fetch ball' -> Selector BT: search -> approach -> ground_pick + quack
    ctrl.client.reset_mock()
    assert ctrl.set_intent("fetch ball") == "FETCH"
    ctrl.world_state.ball_visible = False
    ctrl.world_state.obstacle_close = False
    ctrl.tick_behavior_tree()
    ctrl.client.move.assert_called_with(vx=0.0, vy=0.0, vyaw=0.25)

    ctrl.client.reset_mock()
    ctrl.world_state.ball_visible = True
    ctrl.world_state.obstacle_close = False
    ctrl.tick_behavior_tree()
    ctrl.client.move.assert_called_with(vx=0.12, vy=0.0, vyaw=0.0)

    ctrl.client.reset_mock()
    ctrl.world_state.ball_visible = True
    ctrl.world_state.obstacle_close = True
    ctrl.tick_behavior_tree()
    ctrl.client.do_skill.assert_called_once_with("ground_pick")
    ctrl.client.play_sound.assert_called_once_with("quack")
    assert ctrl.active_intent == "WAIT"

    # 5. 'wait' resets intent cleanly and ambient scan clamps to 0 when unstable
    assert ctrl.set_intent("wait") == "WAIT"
    ctrl.world_state.stability = "LOW"
    ctrl.tick_behavior_tree()
    assert ctrl.head_trim == [0.0, 0.0]

    # 6. Brownout lock stops robot and refuses locomotion
    ctrl.client.reset_mock()
    ctrl.set_intent("patrol")
    ctrl.world_state.update_battery_voltage(6.2)
    ctrl.tick_behavior_tree()
    ctrl.client.stop.assert_called_once()
    ctrl.client.move.assert_not_called()

    # 7. Telemetry staleness watchdog stops robot when telemetry goes silent
    import time
    ctrl.client.reset_mock()
    ctrl.world_state.update_battery_voltage(7.6)
    ctrl.set_intent("come")
    ctrl.telemetry_timeout_s = 0.05
    ctrl.last_telemetry_time = time.monotonic() - 1.0
    ctrl.tick_behavior_tree()
    ctrl.client.stop.assert_called_once()
    ctrl.client.move.assert_not_called()
