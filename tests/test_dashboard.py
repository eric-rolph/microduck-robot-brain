"""Unit tests for the live web telemetry & command dashboard."""

import json
import urllib.request

from microduck_brain.dashboard import DashboardServer
from scripts.run_physical_microduck import PhysicalMicroduckController


def test_dashboard_serves_html_state_sse_and_commands():
    ctrl = PhysicalMicroduckController(dry_run=True)
    ctrl.start()
    ctrl.world_state.update_battery_voltage(7.45)
    ctrl.world_state.update_from_tof_frame({"rows": 8, "cols": 8, "distance_mm": [900] * 64})
    ctrl.tick_behavior_tree()

    srv = DashboardServer(
        snapshot_fn=ctrl.get_telemetry_snapshot,
        command_fn=ctrl.set_intent,
        host="127.0.0.1",
        port=0,
        stream_hz=20.0,
    )
    url = srv.start()
    try:
        # 1. HTML cockpit page
        with urllib.request.urlopen(f"{url}/", timeout=3.0) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "MICRODUCK // DUCKBRAIN COCKPIT" in html

        # 2. JSON state snapshot
        with urllib.request.urlopen(f"{url}/api/state", timeout=3.0) as resp:
            assert resp.status == 200
            snap = json.loads(resp.read().decode("utf-8"))
            assert snap["world_state"]["battery_volts"] == 7.45
            assert len(snap["world_state"]["tof_grid"]) == 64

        # 3. Command dispatch endpoint (natural language -> Tier 1 -> active_intent)
        req = urllib.request.Request(
            f"{url}/api/command",
            data=json.dumps({"command": "patrol"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res == {"ok": True, "intent": "PATROL"}
            assert ctrl.active_intent == "PATROL"

        # 4. E-STOP dispatch
        req_stop = urllib.request.Request(
            f"{url}/api/command",
            data=json.dumps({"command": "stop"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_stop, timeout=3.0) as resp:
            res_stop = json.loads(resp.read().decode("utf-8"))
            assert res_stop == {"ok": True, "intent": "STOP"}
            assert ctrl.active_intent == "STOP"

        # 5. Live SSE stream yields valid event frame
        with urllib.request.urlopen(f"{url}/api/stream", timeout=3.0) as resp:
            assert "text/event-stream" in resp.headers.get("Content-Type", "")
            line = resp.readline().decode("utf-8")
            assert line.startswith("data: ")
            ev = json.loads(line[len("data: "):])
            assert ev["active_intent"] == "STOP"
    finally:
        srv.stop()
        ctrl.stop()
