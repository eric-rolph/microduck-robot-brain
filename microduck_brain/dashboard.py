"""Zero-dependency live web telemetry & command dashboard for DuckBrain.

Serves an offline-ready dark-mode engineering cockpit over HTTP + Server-Sent Events
(stdlib only, no extra packages required on Raspberry Pi CM4):
- Real-time 8x8 ToF depth matrix heatmap
- Active Four-Tier pipeline & Behavior Tree node monitor
- BAM M6 battery voltage sag & body tilt sparklines + safety guard badges
- One-click skills (bow, wave, kick, sit, quack), natural-language command bar, and E-STOP
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any, Callable, Optional

SnapshotFn = Callable[[], dict[str, Any]]
CommandFn = Callable[[str], str]


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Microduck DuckBrain — Live Telemetry Cockpit</title>
<style>
  :root {
    --bg: #0b101b; --panel: #121a2b; --border: #23314d;
    --text: #e8eefc; --muted: #8fa2c4;
    --amber: #fbbf24; --cyan: #38bdf8; --emerald: #34d399;
    --violet: #a78bfa; --red: #f43f5e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
    background: radial-gradient(circle at 80% 0%, #16223a 0%, var(--bg) 55%);
    color: var(--text); min-height: 100vh; padding: 18px;
  }
  .mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
  header {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 12px; padding: 14px 18px; border: 1px solid var(--border);
    border-radius: 14px; background: rgba(18, 26, 43, 0.92); margin-bottom: 16px;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .dot { width: 11px; height: 11px; border-radius: 50%; background: var(--emerald); box-shadow: 0 0 12px var(--emerald); }
  .dot.offline { background: var(--red); box-shadow: 0 0 12px var(--red); }
  h1 { margin: 0; font-size: 18px; letter-spacing: 0.04em; }
  .sub { font-size: 12px; color: var(--muted); }
  .estop {
    background: linear-gradient(135deg, #e11d48, #be123c); color: #fff; border: 2px solid #fda4af;
    border-radius: 10px; padding: 10px 22px; font-weight: 800; letter-spacing: 0.08em;
    cursor: pointer; font-size: 14px; box-shadow: 0 0 18px rgba(244,63,94,0.45);
  }
  .estop:hover { filter: brightness(1.12); }
  .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; }
  .card {
    background: rgba(18, 26, 43, 0.92); border: 1px solid var(--border);
    border-radius: 14px; padding: 16px;
  }
  .card h2 {
    margin: 0 0 12px 0; font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--muted); display: flex; justify-content: space-between;
  }
  .span-4 { grid-column: span 4; }
  .span-5 { grid-column: span 5; }
  .span-6 { grid-column: span 6; }
  .span-7 { grid-column: span 7; }
  .span-8 { grid-column: span 8; }
  .span-12 { grid-column: span 12; }
  @media (max-width: 960px) { .span-4,.span-5,.span-6,.span-7,.span-8 { grid-column: span 12; } }

  .tiers { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  @media (max-width: 720px) { .tiers { grid-template-columns: 1fr 1fr; } }
  .tier {
    border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px;
    background: rgba(11, 16, 27, 0.65);
  }
  .tier .tag { font-size: 11px; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 4px; }
  .tier .val { font-size: 15px; font-weight: 700; word-break: break-word; }
  .t1 { border-color: rgba(251,191,36,0.55); } .t1 .tag { color: var(--amber); }
  .t2 { border-color: rgba(56,189,248,0.55); } .t2 .tag { color: var(--cyan); }
  .t3 { border-color: rgba(167,139,250,0.55); } .t3 .tag { color: var(--violet); }
  .t4 { border-color: rgba(52,211,153,0.55); } .t4 .tag { color: var(--emerald); }

  .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .pill {
    padding: 5px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
    border: 1px solid var(--border); background: #0d1424; color: var(--text);
  }
  .pill.ok { border-color: rgba(52,211,153,0.6); color: var(--emerald); }
  .pill.warn { border-color: rgba(251,191,36,0.65); color: var(--amber); }
  .pill.bad { border-color: rgba(244,63,94,0.8); color: #fda4af; background: rgba(244,63,94,0.16); }

  .tof-grid {
    display: grid; grid-template-columns: repeat(8, 1fr); gap: 4px;
    aspect-ratio: 1 / 1; max-width: 310px; margin: 0 auto;
  }
  .tof-cell {
    border-radius: 5px; display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.92);
    border: 1px solid rgba(255,255,255,0.06);
  }
  .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }
  .metric { background: #0d1424; border: 1px solid var(--border); border-radius: 10px; padding: 10px; }
  .metric .k { font-size: 11px; color: var(--muted); }
  .metric .v { font-size: 22px; font-weight: 800; margin-top: 2px; }
  canvas { width: 100%; height: 145px; background: #0b1120; border-radius: 10px; border: 1px solid var(--border); display: block; }

  .cmd-row { display: flex; gap: 8px; margin-bottom: 10px; }
  .cmd-row input {
    flex: 1; background: #0b1120; border: 1px solid var(--border); border-radius: 9px;
    padding: 10px 12px; color: var(--text); font-size: 14px; outline: none;
  }
  .cmd-row input:focus { border-color: var(--cyan); }
  .btn {
    background: #1a2742; color: var(--text); border: 1px solid #2e426b;
    border-radius: 8px; padding: 8px 13px; font-size: 13px; font-weight: 600; cursor: pointer;
  }
  .btn:hover { background: #233559; border-color: var(--cyan); }
  .btn.primary { background: #0284c7; border-color: #38bdf8; color: #fff; }
  .skills { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
  .log {
    background: #090e1a; border: 1px solid var(--border); border-radius: 9px;
    padding: 10px; height: 138px; overflow-y: auto; font-size: 12px; color: #b7c8e6;
  }
  .log div { padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div id="conn-dot" class="dot"></div>
    <div>
      <h1>MICRODUCK // DUCKBRAIN COCKPIT</h1>
      <div class="sub mono" id="hdr-meta">Connecting to telemetry stream...</div>
    </div>
  </div>
  <button class="estop" onclick="sendCmd('stop')">&#9888; E-STOP</button>
</header>

<div class="grid">
  <div class="card span-12">
    <h2><span>Four-Tier Executive Pipeline</span><span class="mono" id="tick-badge">tick 0</span></h2>
    <div class="tiers">
      <div class="tier t1"><div class="tag mono">TIER 1 // INTENT</div><div class="val mono" id="t1-val">WAIT</div></div>
      <div class="tier t2"><div class="tag mono">TIER 2 // WORLDSTATE</div><div class="val mono" id="t2-val">HIGH / CLEAR</div></div>
      <div class="tier t3"><div class="tag mono">TIER 3 // ACTIVE BT NODE</div><div class="val mono" id="t3-val">AmbientHeadScan</div></div>
      <div class="tier t4"><div class="tag mono">TIER 4 // CMD (vx, vy, wz)</div><div class="val mono" id="t4-val">0.00, 0.00, 0.00</div></div>
    </div>
    <div class="badges mono" id="guard-badges"></div>
  </div>

  <div class="card span-5">
    <h2><span>8&times;8 ToF Depth Matrix</span><span class="mono" id="tof-min">2.00 m clear</span></h2>
    <div class="tof-grid mono" id="tof-grid"></div>
  </div>

  <div class="card span-7">
    <h2><span>BAM M6 Power &amp; Attitude Telemetry</span><span class="mono">6.50 V brownout cutoff</span></h2>
    <div class="metrics mono">
      <div class="metric"><div class="k">BATTERY BUS</div><div class="v" id="m-batt" style="color:var(--emerald)">7.40 V</div></div>
      <div class="metric"><div class="k">BODY TILT</div><div class="v" id="m-tilt" style="color:var(--cyan)">0.0&deg;</div></div>
      <div class="metric"><div class="k">VISUAL TRACK</div><div class="v" id="m-vis" style="color:var(--violet)">OPEN</div></div>
    </div>
    <canvas id="spark" width="640" height="160"></canvas>
  </div>

  <div class="card span-12">
    <h2><span>Natural-Language Intent &amp; One-Click Skills</span><span class="mono" id="cmd-status">ready</span></h2>
    <div class="cmd-row">
      <input id="cmd-input" type="text" placeholder="Type intent (e.g. fetch ball, come, patrol, bow, kick, quack, stop)..." onkeydown="if(event.key==='Enter')submitInput()" />
      <button class="btn primary" onclick="submitInput()">Dispatch</button>
    </div>
    <div class="skills">
      <button class="btn" onclick="sendCmd('fetch ball')">Fetch Ball</button>
      <button class="btn" onclick="sendCmd('come')">Follow</button>
      <button class="btn" onclick="sendCmd('patrol')">Patrol</button>
      <button class="btn" onclick="sendCmd('wait')">Wait / Idle</button>
      <button class="btn" onclick="sendCmd('bow')">Bow</button>
      <button class="btn" onclick="sendCmd('wave')">Wave</button>
      <button class="btn" onclick="sendCmd('kick')">Kick</button>
      <button class="btn" onclick="sendCmd('sit')">Sit</button>
      <button class="btn" onclick="sendCmd('quack')">Quack</button>
    </div>
    <div class="log mono" id="event-log"></div>
  </div>
</div>

<script>
const battHist = [], tiltHist = [], MAX_PTS = 90;
const tofEl = document.getElementById('tof-grid');
for (let i = 0; i < 64; i++) {
  const c = document.createElement('div');
  c.className = 'tof-cell';
  tofEl.appendChild(c);
}

function tofColor(mm) {
  if (!mm || mm <= 10) return 'rgba(51,65,85,0.45)';
  if (mm < 300) return 'rgba(244,63,94,0.88)';
  if (mm < 650) return 'rgba(251,191,36,0.78)';
  if (mm < 1200) return 'rgba(56,189,248,0.58)';
  return 'rgba(52,211,153,0.34)';
}

function drawSpark() {
  const cv = document.getElementById('spark'), ctx = cv.getContext('2d');
  const w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);
  // Brownout line at 6.5V (range 5.5V..8.5V)
  const yCut = h - ((6.5 - 5.5) / 3.0) * (h - 24) - 12;
  ctx.strokeStyle = 'rgba(244,63,94,0.45)'; ctx.setLineDash([6, 6]);
  ctx.beginPath(); ctx.moveTo(0, yCut); ctx.lineTo(w, yCut); ctx.stroke();
  ctx.setLineDash([]);

  const plot = (arr, minV, maxV, color) => {
    if (arr.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.beginPath();
    arr.forEach((v, idx) => {
      const x = (idx / (MAX_PTS - 1)) * (w - 16) + 8;
      const t = Math.max(0, Math.min(1, (v - minV) / (maxV - minV)));
      const y = h - t * (h - 24) - 12;
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  plot(battHist, 5.5, 8.5, '#34d399');
  plot(tiltHist, 0, 35, '#38bdf8');
  ctx.fillStyle = '#8fa2c4'; ctx.font = '11px monospace';
  ctx.fillText('Green: Battery V (5.5-8.5V)   Cyan: Body Tilt (0-35 deg)   Red dash: 6.5V cutoff', 10, 16);
}

function render(s) {
  const ws = s.world_state || {};
  document.getElementById('hdr-meta').textContent =
    `mode=${s.mode || 'walk'}  |  ${s.dry_run ? 'SIM / DRY-RUN' : 'PHYSICAL ROBOTD'}  |  stability=${ws.stability || 'HIGH'}`;
  document.getElementById('tick-badge').textContent = `tick ${s.ticks || 0}`;
  document.getElementById('t1-val').textContent = s.active_target ? `${s.active_intent} (${s.active_target})` : (s.active_intent || 'WAIT');
  document.getElementById('t2-val').textContent = `${ws.stability || 'HIGH'} / ${ws.obstacle_close ? 'OBSTACLE' : 'CLEAR'}`;
  document.getElementById('t3-val').textContent = s.active_node || 'Idle';
  const cmd = s.cmd || {vx:0, vy:0, vyaw:0};
  document.getElementById('t4-val').textContent = `${Number(cmd.vx).toFixed(2)}, ${Number(cmd.vy).toFixed(2)}, ${Number(cmd.vyaw).toFixed(2)}`;

  const badges = [
    ['STABILITY ' + (ws.stability || 'HIGH'), ws.stability === 'HIGH' ? 'ok' : (ws.stability === 'MEDIUM' ? 'warn' : 'bad')],
    [ws.is_brownout_risk ? 'BROWNOUT LOCK' : 'POWER NOMINAL', ws.is_brownout_risk ? 'bad' : 'ok'],
    [ws.is_fallen ? 'FALLEN ABORT' : 'UPRIGHT', ws.is_fallen ? 'bad' : 'ok'],
    [ws.obstacle_close ? 'OBSTACLE < 0.30m' : 'PATH CLEAR', ws.obstacle_close ? 'warn' : 'ok'],
    [ws.target_locked ? `TARGET LOCK #${ws.visual_track_id ?? 0}` : (ws.ball_visible ? 'TARGET VISIBLE' : 'NO TARGET'), ws.target_locked ? 'ok' : ''],
  ];
  document.getElementById('guard-badges').innerHTML = badges.map(([t, c]) => `<span class="pill ${c}">${t}</span>`).join('');

  const grid = ws.tof_grid || [];
  document.getElementById('tof-min').textContent = `${Number(ws.forward_clearance_m ?? 2.0).toFixed(2)} m clear`;
  for (let i = 0; i < 64; i++) {
    const mm = grid[i] ?? 2000;
    const cell = tofEl.children[i];
    cell.style.background = tofColor(mm);
    cell.textContent = mm >= 1000 ? (mm / 1000).toFixed(1) : String(mm);
  }

  const bv = Number(ws.battery_volts ?? 7.4);
  const td = Number(ws.tilt_deg ?? 0.0);
  const battEl = document.getElementById('m-batt');
  battEl.textContent = `${bv.toFixed(2)} V`;
  battEl.style.color = bv <= 6.5 ? 'var(--red)' : (bv <= 6.9 ? 'var(--amber)' : 'var(--emerald)');
  document.getElementById('m-tilt').textContent = `${td.toFixed(1)}°`;
  document.getElementById('m-vis').textContent = ws.target_locked ? `${Number(ws.visual_range_m).toFixed(2)}m` : 'OPEN';

  battHist.push(bv); if (battHist.length > MAX_PTS) battHist.shift();
  tiltHist.push(td); if (tiltHist.length > MAX_PTS) tiltHist.shift();
  drawSpark();

  const evs = s.events || [];
  document.getElementById('event-log').innerHTML = evs.slice().reverse()
    .map(e => `<div>[${e.kind}] ${String(e.message).replace(/</g,'&lt;')}</div>`).join('') || '<div>No events yet.</div>';
}

async function sendCmd(cmd) {
  document.getElementById('cmd-status').textContent = `sending: ${cmd}`;
  try {
    const r = await fetch('/api/command', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({command: cmd})
    });
    const j = await r.json();
    document.getElementById('cmd-status').textContent = j.ok ? `intent -> ${j.intent}` : (j.error || 'error');
  } catch (e) {
    document.getElementById('cmd-status').textContent = 'command failed';
  }
}
function submitInput() {
  const el = document.getElementById('cmd-input');
  if (!el.value.trim()) return;
  sendCmd(el.value.trim());
  el.value = '';
}

function connect() {
  const es = new EventSource('/api/stream');
  es.onopen = () => document.getElementById('conn-dot').classList.remove('offline');
  es.onmessage = (ev) => { try { render(JSON.parse(ev.data)); } catch {} };
  es.onerror = () => document.getElementById('conn-dot').classList.add('offline');
}
fetch('/api/state').then(r => r.json()).then(render).catch(() => {});
connect();
</script>
</body>
</html>
"""


class DashboardServer:
    """Background HTTP + SSE telemetry cockpit for DuckBrain."""

    def __init__(
        self,
        snapshot_fn: SnapshotFn,
        command_fn: Optional[CommandFn] = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        stream_hz: float = 10.0,
    ) -> None:
        self.snapshot_fn = snapshot_fn
        self.command_fn = command_fn
        self.host = host
        self.port = port
        self.stream_interval = 1.0 / max(1.0, stream_hz)
        self._stop_event = threading.Event()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def bound_port(self) -> int:
        if self._server is not None:
            return int(self._server.server_address[1])
        return self.port

    @property
    def url(self) -> str:
        display_host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        return f"http://{display_host}:{self.bound_port}"

    def start(self) -> str:
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return  # Keep robot console clean

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    body = DASHBOARD_HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/api/state":
                    self._send_json(HTTPStatus.OK, parent.snapshot_fn())
                elif self.path == "/api/stream":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    try:
                        while not parent._stop_event.is_set():
                            payload = json.dumps(parent.snapshot_fn())
                            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                            self.wfile.flush()
                            parent._stop_event.wait(parent.stream_interval)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                        return
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

            def do_POST(self) -> None:
                if self.path != "/api/command":
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                    return
                if parent.command_fn is None:
                    self._send_json(HTTPStatus.NOT_IMPLEMENTED, {"ok": False, "error": "read-only mode"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(min(length, 4096)).decode("utf-8")
                    data = json.loads(raw) if raw else {}
                    cmd = str(data.get("command", "")).strip()
                    if not cmd:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "empty command"})
                        return
                    intent = parent.command_fn(cmd)
                    self._send_json(HTTPStatus.OK, {"ok": True, "intent": intent})
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})

        self._stop_event.clear()
        self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        self._stop_event.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
