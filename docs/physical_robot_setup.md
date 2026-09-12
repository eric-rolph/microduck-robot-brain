# Physical Microduck robot setup

Guide for deploying DuckBrain to the physical Microduck biped robot from Pollen Robotics.

## Hardware overview

- **Dimensions**: 25 cm height, 800 g weight
- **Actuation**: 14 Dynamixel XL330 servos (5 per leg, 4 in neck/head) plus 1 beak gripper motor
- **Sensors**: 6-axis IMU on trunk base, 8x8 matrix Time-of-Flight (ToF) range sensor, wide-angle camera, stereo microphones
- **Compute**: Raspberry Pi Compute Module 4 (CM4) running Linux with PREEMPT_RT kernel or real-time priority
- **Daemons**: Pollen `robotd` (motor control, kinematic solver, joint telemetry) and `tofd` (depth sensor stream)

## Architecture on physical hardware

```
User Voice / Text
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ DuckBrain (Python 3.10+)                               │
│  ├─ Tier 1: Intent parser (stateless command parsing) │
│  ├─ Tier 2: WorldState (IMU stability, ToF, battery)   │
│  ├─ Tier 3: Behavior tree (20 Hz executive logic)      │
│  └─ Tier 4: Hardware bridge (JSON-RPC client)          │
└───────────────────────┬────────────────────────────────┘
                        │ JSON-RPC 2.0 (Unix socket or TCP)
       ┌────────────────┴───────────────┐
       ▼                                ▼
┌──────────────┐                 ┌─────────────┐
│ robotd       │                 │ tofd        │
│ /run/robotd  │                 │ /run/tofd   │
│ .sock        │                 │ /tof.sock   │
└──────┬───────┘                 └─────────────┘
       │ Dynamixel Bus (TTL 1 Mbps)
       ▼
 14x XL330 Servos
```

## Deployment modes

### Mode 1: Onboard CM4 execution

DuckBrain runs directly on the Raspberry Pi CM4 alongside `robotd`.

1. Clone repository to `/home/duck/microduck-robot-brain`:
   ```bash
   git clone https://github.com/eric-rolph/microduck-robot-brain.git /home/duck/microduck-robot-brain
   cd /home/duck/microduck-robot-brain
   ```

2. Create virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run pre-flight hardware diagnostics:
   ```bash
   python3 scripts/check_physical_duck.py
   ```

4. Install systemd service for automatic launch on boot:
   ```bash
   sudo cp systemd/microduck-brain.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable microduck-brain.service
   sudo systemctl start microduck-brain.service
   ```

5. Inspect service logs:
   ```bash
   journalctl -u microduck-brain.service -f
   ```

### Mode 2: Remote workstation control over Wi-Fi

Run DuckBrain on your laptop or development workstation while Microduck runs `robotd` on the physical robot.

1. Forward robotd and tofd sockets through SSH:
   ```bash
   ssh -N -L 8088:/run/robotd.sock -L 8089:/run/tofd/tof.sock duck@microduck.local
   ```

2. Run pre-flight check targeting forwarded ports:
   ```bash
   python scripts/check_physical_duck.py --endpoint localhost:8088 --tof-endpoint localhost:8089
   ```

3. Launch interactive control console:
   ```bash
   python scripts/run_physical_microduck.py --endpoint localhost:8088 --tof-endpoint localhost:8089 --interactive
   ```

## Pre-flight hardware checks

Run `scripts/check_physical_duck.py` before untethered runs. The script evaluates 5 gates:

1. **Daemon reachability**: Confirms JSON-RPC communication with `robotd`.
2. **Battery voltage**:
   - Above 7.0V: Optimal (2S LiPo).
   - 6.5V to 7.0V: Warning, recharge recommended.
   - Below 6.5V: Lockout. Locomotion is blocked to prevent brownouts and Dynamixel controller resets.
3. **Safety flags**: Verifies `fallen == false` and `limp == false`.
4. **ToF depth sensor**: Verifies 8x8 matrix distance streaming from `tofd`.
5. **Actuator smoke test**: Executes non-locomotive motion (beak open/close, slight head tilt, quack sound).

To run without physical hardware connected (offline or CI):
```bash
python scripts/check_physical_duck.py --dry-run
```

## Interactive voice and text commands

In interactive mode (`--interactive`), you can issue commands directly at the console prompt:

| Command | Action |
|---|---|
| `come` / `follow` | Enters approach loop, tracks target while avoiding close obstacles |
| `fetch ball` | Scans for ball, approaches target, executes `ground_pick` skill, quacks on success |
| `sit` | Triggers Pollen `sit_toggle` episodic skill |
| `quack` | Plays duck sound through speaker |
| `stop` / `halt` | Deterministic hard emergency stop (zeros all velocities immediately) |
| `status` | Prints live battery voltage, stability trigger, roughness, and forward clearance |
| `exit` | Stops robot and terminates cleanly |

## Safety guardrails

- **Brownout protection**: Monitored through a dual-threshold Schmitt trigger (6.5V low, 6.8V high). If bus voltage sags below 6.5V under load, locomotion is halted immediately.
- **Fall detector**: When `safety.fallen` is reported by the IMU or roll/pitch exceeds 45 degrees, the behavior tree aborts active plans and drops to zero velocity.
- **CPU core isolation**: On the CM4, `systemd/robotd.service` binds the motor loop to Core 4 with FIFO real-time priority (`CPUAffinity=4`, `Nice=-20`). DuckBrain runs on Cores 2-3 (`CPUAffinity=2,3`, `Nice=-5`), guaranteeing inference spikes never induce servo communication jitter.
