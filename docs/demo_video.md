# 1080P Full HD demonstration video

DuckBrain includes a 1080P (1920x1080 @ 30 FPS) physics demonstration video rendered in MuJoCo. It uses the authentic 3D Pollen Robotics Microduck mesh assets and official neural locomotion policies.

## Video summary

- **File**: `output/microduck_brain_demo_1080p.mp4`
- **Resolution**: 1920 x 1080 (Full HD, 16:9)
- **Framerate**: 30.0 FPS (1,080 total frames, 36.0 seconds runtime)
- **Audio**: 44.1 kHz stereo AAC with synthesized motor whine, audio telemetry pings, and quacks
- **Telemetry HUD**: Real-time 50 Hz overlay displaying Tier 1 intent, Tier 2 WorldState debouncing, Tier 3 Behavior Tree states, 61-D observation vector, Dynamixel bus voltage sag, and joint torque telemetry

## Six demonstrated mission scenes

### Scene 1: Multi-modal intent & ToF obstacle detection (0:00 - 0:06)
- **User input**: "Ducky, fetch the marker and bring it back!"
- **Tier 1**: One-shot intent extraction parses `{"action": "FETCH", "target": "marker", "urgency": "HIGH"}` with zero ongoing KV cache.
- **Perception suite**: 8x8 matrix Time-of-Flight (ToF) sensor and camera identify a blocking obstacle directly in the forward path at 0.20 m distance.
- **Tier 3**: Behavior tree activates `SearchActionNode` with 14-DOF biped scanning and 61-D observation vector monitoring.

### Scene 2: ToF detection & autonomous flank avoidance (0:06 - 0:12)
- **Obstacle circumnavigation**: Dynamic behavior tree detects the centered obstacle box ($x = 0.22\,\text{m}, y = 0.00\,\text{m}$) via the 8x8 ToF sensor and plans an evasive flank path.
- **Locomotion**: Microduck steps forward and maneuvers past the obstacle flank with +0.12 m lateral clearance.
- **Decoupled gaze locking**: Head yaw actively counter-rotates to maintain continuous visual tracking of the target marker while the body circumnavigates the obstacle.
- **Physical stability**: Locomotion policy maintains trunk stability with symmetric leg kinematics and zero tipping.

### Scene 3: Articulated beak approach & ground-level marker clamp (0:12 - 0:18)
- **Grounded low-pad approach**: Marker rests on a realistic low-profile desktop pad ($4\,\text{mm}$ height, marker at $z = 0.012\,\text{m}$), completely eliminating artificial pedestals.
- **Whole-body bipedal crouch**: Robot smoothly flexes hips, bends knees, and dorsiflexes ankles, dropping trunk height from $z = 0.120\,\text{m}$ down to $z = 0.055\,\text{m}$.
- **Articulated beak mechanics**: Authentic Pollen Robotics lower jaw (`jaw.stl` + `jaw_soft.stl`) articulates open on the `beak_pitch` hinge, creating a wide $28\,\text{mm}$ aperture ($0.35\,\text{rad}$).
- **Active mesh colliders & clamp**: Active collision geometry (`condim="4"`, $\mu = 1.8$, soft impedance) on upper beak and lower jaw encloses the $14\,\text{mm}$ dry-erase marker. The jaw firmly clamps shut ($0.05\,\text{rad}$) with synchronized audio click and dynamic equality weld engagement.

### Scene 4: Marker payload lift & dynamic stabilization (0:18 - 0:24)
- **Payload lift**: Robot extends legs from the crouch back to full upright standing stance, lifting the $14\,\text{mm}$ marker smoothly from ground level up to $z = 0.24\,\text{m}$ skyward.
- **BAM M6 actuator dynamics**: Motor torque surge draws dynamic current with battery voltage sag monitored in real time on the HUD.
- **Zero-moment point balance**: Head and neck pitch adjust dynamically to counterbalance the forward center-of-mass shift of the held marker.

### Scene 5: Mocap retargeting: 14-DOF Bandai Bow (0:24 - 0:30)
- **Local mocap bridge**: Executes retargeted 14-DOF Bandai Bow trajectory (229 frames @ 50 Hz) translated from parallel motion capture project (`models/mocap/bow_retargeted.npz`).
- **Payload-aware balance**: Head/neck pitch scaled to counterbalance the held marker without forward overbalancing.
- **Dynamic execution**: Microduck performs a deep, courteous bow and returns smoothly to upright stance with zero falling.

### Scene 6: Stable rest sit & mission certified (0:30 - 0:36)
- **User input**: "Ducky, rest and sit."
- **Execution**: Coordinated transition to a stable seated rest posture with marker held securely in beak.
- **Auditor & critic certification**: Zero falls across all 1,080 frames (minimum trunk height $z = 0.055\,\text{m}$ during deliberate squat, max unwanted tilt $8.4^\circ$), all 5 evaluation suite benchmarks passed, officially certified by both Physics Simulation Critic and Cognitive Behavior Critic.

## Rendering the video locally

Render the video with MuJoCo headless rendering and FFmpeg encoding:

```bash
# Render full 1080P Full HD video (generates output/microduck_brain_demo_1080p.mp4)
python scripts/render_demo_video.py
```

Prerequisites:
- `mujoco >= 3.0`
- `ffmpeg` on system PATH
- Official mesh assets in `microduck_brain/sim/mjcf/assets/`
- Trained policy models in `models/`
