# Microduck robot brain

Microduck robot brain separates high-level intent parsing, reactive behavior planning, and low-level motor policy execution for the Microduck quadruped. Motor control, RL policies, and hardware safety run inside `robotd`. The brain selects what to do and which skills to run.

## Why small transformers fail in motor loops

Placing a small (<100 MB, 50M to 100M parameter) sequence model in a closed motor loop creates distinct failure modes:

1. Dynamic replanning vs open-loop generation. An autoregressive sequence like SEARCH -> APPROACH -> PICKUP creates an open-loop token chain. If the target rolls away during approach, the sequence invalidates. Sampling every frame causes policy chattering at state boundaries.
2. Token hallucination and precondition violations. Small models lack physical grounding. A 50M model will generate PICKUP when an object is not visible simply because the user prompt contained "fetch the ball".
3. Sensor noise and attention resets. Quadruped walking creates IMU vibration and camera bounce. Noisy boolean tokens reset attention weights and cause abandoned tasks.
4. Compute contention. At <100 MB, models lack emergent reasoning yet compete with 50 to 100 Hz motor loops for memory bus bandwidth and CPU caches.
5. Long-horizon drift. Ongoing autoregressive execution bloats KV caches and drifts into repetitive action loops.

The architecture solves this by restricting the Transformer to one-shot speech parsing at the user boundary, delegating reactive execution to a Behavior Tree, and running motor policies inside an isolated real-time daemon.

## Four-tier pipeline

```
[ Speech / User input ]
          |
          v
+-------------------------------------------------------------+
| Tier 1: Intent extraction (stateless transformer / SLM)     |
| Extracts typed JSON goal once, drops context                |
+------------------------------+------------------------------+
                               | ParsedIntent
                               v
+-------------------------------------------------------------+
| Tier 2: WorldState sanitization (10 to 20 Hz)               |
| Schmitt trigger hysteresis and temporal majority voting     |
+------------------------------+------------------------------+
                               | Debounced tokens
                               v
+-------------------------------------------------------------+
| Tier 3: Behavior tree (10 to 20 Hz)                         |
| Dynamic sequences, selectors, and ambient parallel blending |
| Expressions strictly subordinate to gait balance            |
+------------------------------+------------------------------+
                               | Kinematic command (vx, vy, wz, roll, pitch)
                               v
+-------------------------------------------------------------+
| Tier 4: robotd execution (50 to 100 Hz real-time)           |
| Core-pinned SCHED_FIFO loop, POSIX shared memory (SeqLock)  |
| Attitude command filter -> 48D ONNX policy -> 12 leg PD     |
+-------------------------------------------------------------+
```

### Tier 1: Stateless intent extraction
Runs once per command string. It outputs a typed goal and exits:

```json
{
  "action": "FETCH",
  "target": "ball",
  "urgency": "MEDIUM"
}
```

The model maintains no ongoing KV cache.

### Tier 2: WorldState sanitization
Converts continuous noisy sensor data into stable discrete tokens:
* Schmitt trigger hysteresis: IMU pitch and roll variance transitions to `LOW` stability above 0.65 rad/s^2, returning to `HIGH` only after falling below 0.35 rad/s^2.
* Temporal debouncing: Vision detections require positive identification in 7 of the last 10 frames before asserting visibility.

### Tier 3: Reactive behavior tree
Manages task progression and body language:
* Sequential actions: Steps through Search, Approach, Pickup, and Return.
* Asymmetric parallel node: Ticks locomotion while running ambient head scanning. If stability drops, scanning offsets clamp to zero.
* Failure expressions: Single failures trigger an inquisitive head tilt. Three consecutive failures trigger a head shake. If balance drops, expressions are suppressed.

### Tier 4: robotd and motor execution
Communicates with Tier 3 over lock-free POSIX shared memory (`SeqLockChannel`) with single-digit nanosecond latency.
* Process isolation: Core 4 dedicated to the 50 to 100 Hz loop via `SCHED_FIFO` priority 99.
* Attitude command filter: Clamps body lean rate of change to 0.5 rad/s and attenuates lean to zero as forward speed approaches 0.8 m/s.
* Locomotion policy: 48D observation vector feeds an ONNX actor network, commanding 12 joint PD targets.

## Repository layout

```
microduck-robot-brain/
├── CMakeLists.txt              # C++20 build config for robotd and bt_bridge
├── include/
│   ├── duck_shm.hpp            # Aligned shared memory structs and SeqLock
│   └── shm_client.hpp          # POSIX shm manager with Windows fallback
├── src/
│   ├── robotd_main.cpp         # 50 Hz real-time motor daemon
│   └── bt_bridge_node.cpp      # 20 Hz Behavior Tree IPC client
├── microduck_brain/            # Python core package
│   ├── intent_parser.py        # Stateless goal extractor
│   ├── world_state.py          # Schmitt triggers and debouncers
│   ├── logit_masking.py        # Precondition logit masking
│   ├── skill_latch.py          # Asynchronous skill latching
│   ├── behavior_tree.py        # BT nodes and expression logic
│   ├── locomotion_engine.py    # Attitude filter and ONNX runner
│   ├── pollen_bridge.py        # IPC client for Pollen robotd socket
│   └── sim/                    # Sim-to-real MuJoCo physics stack
│       ├── mjcf/               # Self-contained Microduck XML models
│       ├── bam_actuator.py     # BAM M6 DC motor & battery sag model
│       ├── backlash.py         # Mechanical backlash twin dynamics
│       ├── env.py              # Gym environment with 61D contract
│       └── train_smoke.py      # Local PyTorch PPO training runner
├── isaac_lab/
│   ├── env_cfg.py              # Isaac Lab RL config with domain randomization
│   └── export_onnx.py          # ONNX model exporter
├── systemd/
│   ├── robotd.service          # Real-time systemd service unit
│   ├── realtime.slice          # Cgroup v2 slice for Core 4
│   └── 10-cpu-affinity.conf    # System-wide CPU shielding
├── sysctl/
│   └── 99-realtime.conf        # Kernel latency tuning
├── scripts/
│   ├── shield_rt_cores.sh      # IRQ affinity steering script
│   ├── plot_latency.py         # Cyclictest latency distribution plotter
│   ├── run_simulation.py       # End-to-end multi-tier simulation demo
│   └── run_mujoco_sim.py       # Behavior Tree MuJoCo sim runner
└── tests/                      # Pytest unit & integration test suite
```

## Quick start

### Python package and tests

Install dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
python -m pytest tests/ -v
```

Run simulation:
```bash
python scripts/run_simulation.py
```

### Build C++ real-time daemon

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

Run daemon and bridge:
```bash
# Terminal 1: Real-time motor daemon (run with RT priority on Linux)
./robotd

# Terminal 2: Behavior tree bridge node
./bt_bridge_node
```

## Real-time Linux kernel setup

1. Copy kernel sysctl configuration:
```bash
sudo cp sysctl/99-realtime.conf /etc/sysctl.d/
sudo sysctl --system
```

2. Configure systemd CPU shielding and real-time slice:
```bash
sudo cp systemd/10-cpu-affinity.conf /etc/systemd/system.conf.d/
sudo cp systemd/realtime.slice /etc/systemd/system/
sudo cp systemd/robotd.service /etc/systemd/system/
sudo systemctl daemon-reload
```

3. Mask hardware IRQs away from Core 4:
```bash
sudo chmod +x scripts/shield_rt_cores.sh
sudo ./scripts/shield_rt_cores.sh
```

4. Boot kernel flags in `/etc/default/grub`:
```text
GRUB_CMDLINE_LINUX_DEFAULT="... isolcpus=4 nohz_full=4 rcu_nocbs=4"
```
Run `sudo update-grub` and reboot.

5. Start robotd service:
```bash
sudo systemctl enable --now robotd.service
```

## Latency verification

Verify scheduling latency under stress using `cyclictest`:

```bash
# Terminal 1: Background load on housekeeping cores (0-3, 5-7)
sudo stress-ng --taskset 0-3,5-7 --cpu 7 --vm 2 --vm-bytes 1G --io 4 --timeout 10m

# Terminal 2: Measurement thread on Core 4
sudo cyclictest --mlockall --priority=99 --affinity=4 --mainaffinity=0 \
  --threads=1 --interval=200 --duration=10m --histogram=100 --histfile=latency_hist.txt
```

Plot results:
```bash
python scripts/plot_latency.py latency_hist.txt
```

## Sim-to-real MuJoCo pipeline

Pollen Robotics trains Microduck locomotion and recovery behaviors in MuJoCo and deploys identical weights directly to physical bipeds. The framework bridges the reality gap with actuator physics, mechanical backlash emulation, and a strict 61-D observation contract.

### 61-D observation contract

```
[ Proprioception: 48D ]
  0..3:   Base angular velocity from IMU gyro (rad/s)
  3..6:   Projected gravity vector in trunk frame (unit vector)
  6..20:  Joint positions relative to HOME_FRAME through backlash (rad)
  20..34: Joint velocities through backlash (rad/s)
  34..48: Last applied action (14D)

[ Commands: 13D ]
  48..51: Twist command [vx, vy, vyaw] (m/s, m/s, rad/s)
  51..55: Head pose delta [neck_pitch, head_pitch, head_yaw, head_roll] (rad)
  55..61: Body pose delta [x, y, z, roll, pitch, yaw] (m, m, m, rad, rad, rad)
Total: 48 + 13 = 61 dimensions
```

### BAM M6 actuator modeling
Dynamixel XL330 servos exhibit non-linear behavior under load:
* Battery voltage sag: Battery voltage drops from 8.2 V down to ~6.5 V under surge currents, reducing available torque: `V_batt = max(V_min, V_oc - I_total * R_internal)`.
* Back-EMF velocity limit: Torque decreases as motor velocity rises: `V_eff = max(0, V_batt - Ke * |vel|)`.
* Gearbox friction: XL330 gear trains introduce Coulomb friction and stiction thresholds that prevent motion under sub-threshold torques.
* Transport delay: Models digital bus communication delay (1 to 4 steps) before commands reach the motors.

### Mechanical backlash twin
To emulate 3D-printed horn play and gear backlash, `microduck_walk_backlash.xml` pairs every actuated joint with an unactuated `passive_<joint>_backlash` hinge in series (+/-1 degree free play). The simulation passes output-side encoder readings `q_enc = q_servo + q_backlash` into the 61-D observation vector, forcing policies to stabilize across mechanical deadbands.

### Local PPO training run

Run a local PyTorch PPO training run on CPU or CUDA:

```bash
# Smoke test (5 iterations, 4 parallel environments)
python -m microduck_brain.sim.train_smoke --iterations 5 --num-envs 4 --steps-per-env 64

# Export trained policy weights directly to ONNX
python -m microduck_brain.sim.train_smoke --iterations 10 --export-onnx models/microduck_walk.onnx
```

### End-to-end DuckBrain MuJoCo simulation

Execute high-level Behavior Tree goals (`FETCH_BALL` -> Search in-place turn -> Approach forward walk -> Pickup ground dip) driving the 50 Hz physics loop:

```bash
# Headless run with cycle latency and battery sag summary
python scripts/run_mujoco_sim.py --steps 250 --onnx models/microduck_walk.onnx --headless

# Interactive viewer mode (requires display)
python scripts/run_mujoco_sim.py --steps 250 --onnx models/microduck_walk.onnx --render
```

## Isaac Lab policy export

Export trained checkpoints to standalone ONNX:
```bash
python isaac_lab/export_onnx.py --checkpoint logs/model.pt --output microduck_locomotion.onnx
```

## Physical Microduck robot deployment

DuckBrain connects directly to Pollen Robotics physical Microduck biped robot over Unix domain sockets (`/run/robotd.sock`, `/run/tofd/tof.sock`) or across a TCP network tunnel.

### Pre-flight hardware checks

Before running untethered, run the pre-flight verification script to test daemon connectivity, battery voltage (> 6.5V brownout limit), safety state, and sensor feeds:

```bash
# On physical robot (or over SSH tunnel):
python scripts/check_physical_duck.py

# Offline verification / CI dry-run:
python scripts/check_physical_duck.py --dry-run
```

### Interactive brain execution

Launch the 20 Hz executive brain with an interactive natural-language console:

```bash
# Onboard CM4:
python scripts/run_physical_microduck.py --interactive

# Remote workstation over Wi-Fi (SSH forwarded sockets):
ssh -N -L 8088:/run/robotd.sock -L 8089:/run/tofd/tof.sock duck@microduck.local &
python scripts/run_physical_microduck.py --endpoint localhost:8088 --tof-endpoint localhost:8089 --interactive
```

Type commands at the prompt:
```text
ducky> come
ducky> fetch ball
ducky> sit
ducky> quack
ducky> status
ducky> stop
```

For full systemd setup, brownout thresholds, and real-time core isolation details, see [docs/physical_robot_setup.md](docs/physical_robot_setup.md).

## License

Apache-2.0

