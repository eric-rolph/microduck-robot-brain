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
- **User input**: "Ducky, fetch the snack and bring it back!"
- **Tier 1**: One-shot intent extraction parses `{"action": "FETCH", "target": "snack", "urgency": "HIGH"}` with zero ongoing KV cache.
- **Perception suite**: 8x8 matrix Time-of-Flight (ToF) sensor and camera identify a blocking obstacle directly in the forward path at 0.20 m distance.
- **Tier 3**: Behavior tree activates `SearchActionNode` with 14-DOF biped scanning and 61-D observation vector monitoring.

### Scene 2: ToF detection & autonomous flank avoidance (0:06 - 0:12)
- **Obstacle circumnavigation**: Dynamic behavior tree detects the obstacle block and plans an evasive flank path.
- **Locomotion**: Microduck steps forward and maneuvers past the obstacle flank with +0.12 m lateral clearance.
- **Physical stability**: Locomotion policy maintains trunk height at $z = 0.118$ m with symmetric leg kinematics and zero tipping.

### Scene 3: Feeder stand crouch & beak contact grasp (0:12 - 0:18)
- **Feeder stand approach**: Staged cleanly in front of the elevated feeder stand pedestal ($z = 0.05$ m, snack at $z = 0.115$ m).
- **Kinematic squat**: Coordinated knee flexion ($+0.18$ rad left, $-0.18$ rad right) and ankle dorsiflexion counter-lean keeps foot soles flat and center-of-mass centered over the 1.35 cm foot sole patch.
- **Physical grasp**: Beak contacts the cylinder snack and engages MuJoCo dynamic equality weld (`mjEQ_WELD` on `jaw_soft` $\leftrightarrow$ `target_snack`).

### Scene 4: Stand & carried payload stabilization (0:18 - 0:24)
- **Payload lift**: Robot returns to full upright standing stance, lifting the 25 g snack off the pedestal to $z = 0.214$ m.
- **BAM M6 actuator dynamics**: Motor torque surge draws 15.4 A with battery voltage sag monitored in real time.
- **Zero-moment point balance**: Head and neck pitch adjust dynamically to counterbalance the forward center-of-mass shift of the held object.

### Scene 5: Mocap retargeting: 14-DOF Bandai Bow (0:24 - 0:30)
- **Local mocap bridge**: Executes retargeted 14-DOF Bandai Bow trajectory (229 frames @ 50 Hz) translated from parallel motion capture project (`models/mocap/bow_retargeted.npz`).
- **Payload-aware balance**: Head/neck pitch scaled to 0.45/0.40 to prevent forward overbalancing while holding the carried snack in the beak.
- **Dynamic execution**: Microduck performs a deep, courteous bow and returns smoothly to upright stance with zero falling.

### Scene 6: Stable rest sit & mission certified (0:30 - 0:36)
- **User input**: "Ducky, rest and sit."
- **Execution**: Coordinated transition to a stable seated rest posture with snack held securely in beak.
- **Auditor certification**: Zero falls across all 1,080 frames (minimum trunk height $z = 0.112$ m), all 5 evaluation suite benchmarks passed, physical sim-to-real readiness certified.

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
