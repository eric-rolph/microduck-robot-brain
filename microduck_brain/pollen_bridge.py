"""
Pollen Robotics robotd JSON-RPC socket client bridge.
Connects DuckBrain (Tier 1-3) directly to Pollen's /run/robotd.sock and /run/tofd/tof.sock.
"""

from __future__ import annotations
import json
import socket
import threading
from typing import Any, Callable, Optional


class PollenRobotdClient:
    """
    JSON-RPC 2.0 client for Pollen Robotics' robotd daemon over Unix domain socket.
    Handles continuous notification streams (robot.move, robot.head)
    and discrete requests (robot.do, robot.stop, robot.sound).
    """

    def __init__(self, socket_path: str = "/run/robotd.sock") -> None:
        self.socket_path = socket_path
        self.sock: Optional[socket.socket] = None
        self.msg_id = 0
        self.lock = threading.Lock()
        self.last_state: dict[str, Any] = {}

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.socket_path)
            return True
        except (OSError, FileNotFoundError):
            self.sock = None
            return False

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def send_notification(self, method: str, params: dict[str, Any]) -> bool:
        """Sends continuous intent as JSON-RPC notification (no id, no reply)."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        return self._send_raw(payload)

    def send_request(self, method: str, params: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        """Sends discrete call and waits for answer."""
        with self.lock:
            self.msg_id += 1
            call_id = self.msg_id

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": call_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        if not self._send_raw(payload):
            return None

        # Read single line response
        if self.sock is None:
            return None
        try:
            response_data = b""
            while not response_data.endswith(b"\n"):
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
            if response_data:
                return json.loads(response_data.decode("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return None

    def move(self, vx: float, vy: float, vyaw: float) -> bool:
        """Continuous body velocity twist in m/s and rad/s."""
        return self.send_notification("robot.move", {"vx": vx, "vy": vy, "vyaw": vyaw})

    def head(self, neck_pitch: float, head_pitch: float, head_yaw: float, head_roll: float) -> bool:
        """Continuous head and neck angles in radians."""
        return self.send_notification("robot.head", {
            "neck_pitch": neck_pitch,
            "head_pitch": head_pitch,
            "head_yaw": head_yaw,
            "head_roll": head_roll,
        })

    def look(self, x: float, y: float, z: float) -> Optional[dict[str, Any]]:
        """Discrete gaze point in trunk frame meters; robotd runs inverse kinematics."""
        return self.send_request("robot.look", {"point": [x, y, z]})

    def do_skill(self, skill_name: str) -> Optional[dict[str, Any]]:
        """Triggers episodic skill (e.g. ground_pick, kick, sit_toggle)."""
        return self.send_request("robot.do", {"skill": skill_name})

    def stop(self) -> Optional[dict[str, Any]]:
        """Zeros velocities immediately."""
        return self.send_request("robot.stop")

    def play_sound(self, sound_tag: str) -> Optional[dict[str, Any]]:
        """Plays sound tag from voice-bank (e.g. quack)."""
        return self.send_request("robot.sound", {"sound": sound_tag})

    def mouth(self, open_ratio: float) -> bool:
        """Commands beak opening (0.0 closed to 1.0 open)."""
        return self.send_notification("robot.mouth", {"open": open_ratio})

    def _send_raw(self, payload: dict[str, Any]) -> bool:
        if self.sock is None:
            return False
        try:
            data = (json.dumps(payload) + "\n").encode("utf-8")
            self.sock.sendall(data)
            return True
        except OSError:
            self.close()
            return False
