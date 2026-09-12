"""
Pollen Robotics robotd and tofd JSON-RPC socket client bridge.
Connects DuckBrain (Tier 1-3) directly to Pollen's /run/robotd.sock and /run/tofd/tof.sock,
or across network TCP endpoints for remote workstation control.
"""

from __future__ import annotations
import json
import socket
import threading
import time
from typing import Any, Callable, Optional, Tuple


def parse_endpoint(endpoint: str) -> Tuple[str, Optional[Tuple[str, int]]]:
    """
    Parses an endpoint into socket family and address tuple.
    Supports:
      - Unix socket path: "/run/robotd.sock"
      - TCP address: "192.168.1.50:8088", "localhost:8088", "tcp://microduck.local:8088"
    """
    cleaned = endpoint.strip()
    if cleaned.startswith("tcp://"):
        cleaned = cleaned[6:]

    # Check for host:port (ensuring not a Windows drive letter like C:\...)
    if ":" in cleaned and not (len(cleaned) >= 2 and cleaned[1] == ":" and cleaned[2] in ("\\", "/")):
        host, port_str = cleaned.rsplit(":", 1)
        return "tcp", (host, int(port_str))

    return "unix", None


class PollenRobotdClient:
    """
    JSON-RPC 2.0 client for Pollen Robotics robotd daemon.
    Supports Unix domain sockets, TCP network sockets, demultiplexed request-reply,
    and continuous telemetry subscription streams.
    """

    def __init__(self, endpoint: str = "/run/robotd.sock") -> None:
        self.endpoint = endpoint
        self.sock: Optional[socket.socket] = None
        self.msg_id = 0
        self.send_lock = threading.Lock()
        self.pending_responses: dict[int, dict[str, Any]] = {}
        self.response_events: dict[int, threading.Event] = {}
        self.pending_lock = threading.Lock()

        self.last_state: dict[str, Any] = {}
        self.state_callback: Optional[Callable[[dict[str, Any]], None]] = None

        self._running = False
        self._reader_thread: Optional[threading.Thread] = None

    @property
    def is_connected(self) -> bool:
        return self.sock is not None and self._running

    def connect(self, timeout: float = 2.0) -> bool:
        """Connect to robotd over Unix socket or TCP."""
        self.close()
        family, tcp_addr = parse_endpoint(self.endpoint)

        try:
            if family == "tcp" and tcp_addr is not None:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect(tcp_addr)
                s.settimeout(None)
                self.sock = s
            else:
                if not hasattr(socket, "AF_UNIX"):
                    return False
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect(self.endpoint)
                s.settimeout(None)
                self.sock = s

            self._running = True
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            return True
        except (OSError, ValueError):
            self.sock = None
            self._running = False
            return False

    def close(self) -> None:
        """Closes socket connection and stops receiver thread."""
        self._running = False
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

        if self._reader_thread is not None and self._reader_thread.is_alive():
            if threading.current_thread() != self._reader_thread:
                self._reader_thread.join(timeout=0.5)
            self._reader_thread = None

        with self.pending_lock:
            for event in self.response_events.values():
                event.set()
            self.response_events.clear()
            self.pending_responses.clear()

    def _reader_loop(self) -> None:
        """Background thread reading newline-delimited JSON-RPC messages."""
        buffer = b""
        while self._running and self.sock is not None:
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue
                    self._handle_incoming_message(line_str)
            except (OSError, ValueError):
                break

        self._running = False

    def _handle_incoming_message(self, line: str) -> None:
        """Routes responses by id and server notifications by method."""
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return

        if not isinstance(msg, dict):
            return

        # Handle request-response
        if "id" in msg and msg["id"] is not None:
            req_id = msg["id"]
            with self.pending_lock:
                self.pending_responses[req_id] = msg
                event = self.response_events.get(req_id)
                if event is not None:
                    event.set()
            return

        # Handle server notifications (e.g. robot.state)
        method = msg.get("method")
        params = msg.get("params", {})
        if method == "robot.state":
            self.last_state = params
            if self.state_callback is not None:
                try:
                    self.state_callback(params)
                except Exception:
                    pass

    def send_notification(self, method: str, params: dict[str, Any]) -> bool:
        """Sends continuous intent as JSON-RPC notification (no id, no reply expected)."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        return self._send_raw(payload)

    def send_request(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        timeout: float = 2.0,
    ) -> Optional[dict[str, Any]]:
        """Sends discrete call and waits for answer matching request id."""
        with self.send_lock:
            self.msg_id += 1
            call_id = self.msg_id

        event = threading.Event()
        with self.pending_lock:
            self.response_events[call_id] = event

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": call_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        if not self._send_raw(payload):
            with self.pending_lock:
                self.response_events.pop(call_id, None)
            return None

        # If reader loop is not running, fallback to synchronous receive
        if not self._running or self._reader_thread is None:
            return self._sync_recv(call_id, timeout)

        # Wait for reader thread to notify
        got_reply = event.wait(timeout=timeout)
        with self.pending_lock:
            self.response_events.pop(call_id, None)
            return self.pending_responses.pop(call_id, None) if got_reply else None

    def _sync_recv(self, call_id: int, timeout: float) -> Optional[dict[str, Any]]:
        """Fallback synchronous reader when reader loop is inactive."""
        if self.sock is None:
            return None
        self.sock.settimeout(timeout)
        try:
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
            if buf:
                msg = json.loads(buf.decode("utf-8"))
                if msg.get("id") == call_id:
                    return msg
        except (OSError, json.JSONDecodeError):
            return None
        finally:
            if self.sock is not None:
                self.sock.settimeout(None)
        return None

    def subscribe_state(self, callback: Callable[[dict[str, Any]], None]) -> bool:
        """Subscribes to 50 Hz robot.state telemetry stream."""
        self.state_callback = callback
        resp = self.send_request("robot.subscribe")
        return resp is not None and resp.get("result", {}).get("subscribed", True)

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

    def pose(self, z: float, roll: float, pitch: float, active: bool = True) -> bool:
        """Continuous standing posture trims (z in m, roll/pitch in radians)."""
        return self.send_notification("robot.pose", {
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "active": active,
        })

    def look(self, x: float, y: float, z: float) -> Optional[dict[str, Any]]:
        """Discrete gaze point in trunk frame meters."""
        return self.send_request("robot.look", {"point": [x, y, z]})

    def do_skill(self, skill_name: str) -> Optional[dict[str, Any]]:
        """Triggers episodic skill (e.g. ground_pick, sit_toggle, kick_left)."""
        return self.send_request("robot.do", {"skill": skill_name})

    def stop(self) -> Optional[dict[str, Any]]:
        """Zeros velocities immediately."""
        return self.send_request("robot.stop")

    def play_sound(self, sound_tag: str) -> Optional[dict[str, Any]]:
        """Plays sound tag from soundbank (e.g. quack)."""
        return self.send_request("robot.sound", {"sound": sound_tag})

    def mouth(self, open_ratio: float) -> bool:
        """Commands beak opening (0.0 closed to 1.0 open)."""
        return self.send_notification("robot.mouth", {"open": open_ratio})

    def get_mode(self) -> Optional[str]:
        """Queries current locomotion mode ('walk' or 'roller')."""
        resp = self.send_request("robot.mode")
        if resp and "result" in resp:
            return resp["result"].get("mode")
        return None

    def set_mode(self, mode: str) -> Optional[dict[str, Any]]:
        """Switches drive mode ('walk' or 'roller')."""
        return self.send_request("robot.setMode", {"mode": mode})

    def get_skills(self) -> Optional[dict[str, Any]]:
        """Queries list of registered skills."""
        return self.send_request("robot.skills")

    def get_model(self) -> Optional[dict[str, Any]]:
        """Queries robot kinematic model and joint structure."""
        return self.send_request("robot.model")

    def shutdown(self) -> Optional[dict[str, Any]]:
        """Commands robot to sit and safely power down."""
        return self.send_request("robot.shutdown")

    def _send_raw(self, payload: dict[str, Any]) -> bool:
        if self.sock is None:
            return False
        try:
            data = (json.dumps(payload) + "\n").encode("utf-8")
            with self.send_lock:
                self.sock.sendall(data)
            return True
        except OSError:
            self.close()
            return False


class PollenTofClient:
    """
    JSON-RPC client for Pollen Robotics tofd daemon.
    Streams 8x8 matrix distance frames and head IMU samples.
    """

    def __init__(self, endpoint: str = "/run/tofd/tof.sock") -> None:
        self.endpoint = endpoint
        self.sock: Optional[socket.socket] = None
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None

        self.last_tof_frame: dict[str, Any] = {}
        self.last_imu_frame: dict[str, Any] = {}
        self.tof_callback: Optional[Callable[[dict[str, Any]], None]] = None
        self.imu_callback: Optional[Callable[[dict[str, Any]], None]] = None

    def connect(self, timeout: float = 2.0) -> bool:
        self.close()
        family, tcp_addr = parse_endpoint(self.endpoint)

        try:
            if family == "tcp" and tcp_addr is not None:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect(tcp_addr)
                s.settimeout(None)
                self.sock = s
            else:
                if not hasattr(socket, "AF_UNIX"):
                    return False
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect(self.endpoint)
                s.settimeout(None)
                self.sock = s

            self._running = True
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

            # Subscribe to tof stream
            sub_payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tof.subscribe"}) + "\n"
            self.sock.sendall(sub_payload.encode("utf-8"))
            return True
        except (OSError, ValueError):
            self.sock = None
            self._running = False
            return False

    def close(self) -> None:
        self._running = False
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        if self._reader_thread is not None and self._reader_thread.is_alive():
            if threading.current_thread() != self._reader_thread:
                self._reader_thread.join(timeout=0.5)
            self._reader_thread = None

    def _reader_loop(self) -> None:
        buffer = b""
        while self._running and self.sock is not None:
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue
                    try:
                        msg = json.loads(line_str)
                    except json.JSONDecodeError:
                        continue

                    method = msg.get("method")
                    params = msg.get("params", {})
                    if method == "tof.frame":
                        self.last_tof_frame = params
                        if self.tof_callback is not None:
                            self.tof_callback(params)
                    elif method == "head_imu.frame":
                        self.last_imu_frame = params
                        if self.imu_callback is not None:
                            self.imu_callback(params)
            except (OSError, ValueError):
                break
        self._running = False
