# 1080P Full HD demonstration video

DuckBrain includes a 1080P (1920x1080 @ 30 FPS) physics demonstration video rendered in MuJoCo. It uses the authentic 3D Pollen Robotics Microduck mesh assets and official neural locomotion policies.

## Video summary

- **File**: `output/microduck_brain_demo_1080p.mp4`
- **Resolution**: 1920 x 1080 (Full HD, 16:9)
- **Framerate**: 30.0 FPS (1,080 total frames, 36.0 seconds runtime)
- **Audio**: 44.1 kHz stereo AAC with synthesized motor whine, audio telemetry pings, and quacks
- **Telemetry HUD**: Real-time 50 Hz overlay displaying Tier 1 intent, Tier 2 WorldState debouncing, Tier 3 Behavior Tree states, 61-D observation vector, Dynamixel bus voltage sag, and joint torque telemetry

## Six demonstrated mission scenes

### Scene 1: Intent parsing and target search (0:00 - 0:06)
- **User input**: "Ducky, bring me the ball."
- **Tier 1**: One-shot intent extraction parses `FETCH_BALL` goal with zero ongoing KV cache.
- **Tier 3**: Behavior tree activates `SearchActionNode`. The 14-DOF biped executes coordinated yaw scanning with sagittal center-of-mass counter-balancing (`hip_pitch` lean) to keep the zero-moment point inside the 1.35 cm foot sole contact patch.
- **Tier 2**: Vision detector registers the yellow ball across 5 consecutive frames, transitioning the state machine.

### Scene 2: Approach locomotion with vision debouncing (0:06 - 0:12)
- **Policy**: Direct execution of Pollen's `alpha_walking.onnx` policy.
- **Locomotion**: Biped advances 12.5 cm across the floor toward the ball.
- **Symmetry handling**: Joint kinematics properly map inverted sign conventions across left and right leg pitch, knee, and ankle joints.
- **Stability**: Trunk height holds steady at 0.118 m with zero tipping.

### Scene 3: Dynamic disturbance rejection on rough terrain (0:12 - 0:18)
- **Disturbance**: External push impulse injected directly into the robot trunk.
- **Tier 2 Schmitt trigger**: Fast attitude filter detects impulse and gates balance defense.
- **Recovery**: Active ankle and hip compliance absorbs the energy, restoring upright equilibrium with maximum tilt under 5.0 degrees.
- **Zero falls**: Non-foot collision watchdog confirms zero ground strikes by knees, trunk, or head.

### Scene 4: Ground pickup and BAM M6 motor current sag (0:18 - 0:24)
- **Kinematics**: Deep crouch with knee flexion and neck pitch depression brings the beak to floor level (`z = 0.035 m`) at the ball.
- **BAM M6 actuator model**: High joint torques demand 16.0 A total bus current, producing simulated battery voltage sag from 7.4 V down to 6.30 V.
- **Hysteresis defense**: WorldState brownout lock engages, preventing servo chattering while holding stance.

### Scene 5: Emergency stop and inquisitive head tilt (0:24 - 0:30)
- **Obstacle detection**: Red obstacle block triggers proximity warning.
- **Deterministic halt**: Tier 4 bypasses neural policy to trigger immediate 0.0 m/s stance lock.
- **Behavior tree failure branch**: On blocked path, Tier 3 executes an inquisitive 18-degree head tilt gesture without disturbing trunk balance.

### Scene 6: Rest posture and mission completion (0:30 - 0:36)
- **User input**: "Ducky, rest and sit."
- **Execution**: Coordinated transition to stable seated rest posture.
- **Auditor certification**: Displays passed status across all 5 benchmark tasks and physical certification criteria.

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
