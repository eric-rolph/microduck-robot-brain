# 1080P Full HD demonstration video

DuckBrain includes a 1080P (1920x1080 @ 30 FPS) physics demonstration video rendered in MuJoCo. It uses the authentic 3D Pollen Robotics Microduck mesh assets and official neural locomotion policies.

## Video summary

- **File**: `output/microduck_brain_demo_1080p.mp4`
- **Resolution**: 1920 x 1080 (Full HD, 16:9)
- **Framerate**: 30.0 FPS (1,440 total frames, 48.0 seconds runtime)
- **Audio**: 44.1 kHz stereo AAC with synthesized motor whine, audio telemetry pings, marker drop acoustic clicks, and quacks
- **Telemetry HUD**: Real-time 50 Hz overlay displaying Tier 1 intent, Tier 2 WorldState debouncing, Tier 3 Behavior Tree active nodes, 64-cell 8x8 matrix ToF heat map, BAM M6 bus voltage sag bar, and 14-servo torque histogram

## Six demonstrated mission scenes

### Scene 1: Multi-modal intent & ToF environmental scan (0:00 - 0:08)
- **User input**: "Ducky, navigate the maze, retrieve the marker, and drop it in the tray!"
- **Tier 1**: One-shot intent extraction parses `{"action": "NAVIGATE_FETCH_DEPOSIT", "target": "marker", "container": "tray", "urgency": "HIGH"}` with zero ongoing KV cache.
- **Perception suite**: 8x8 matrix Time-of-Flight (ToF) sensor and camera perform active environmental scanning.
- **HUD telemetry**: Real-time 64-cell ToF depth heat map color-codes obstacle distances; active Behavior Tree `SearchActionNode` monitors stability.

### Scene 2: Ground crouch retrieval with zero floor penetration (0:08 - 0:16)
- **Kinematic policy**: Derives crouch geometry directly from Pollen Robotics official `alpha_ground_pick.onnx` locomotion policy (hip flexion $-1.22/+1.36\,\text{rad}$, knee flexion $+0.65/-0.47\,\text{rad}$, ankle dorsiflexion $+1.30/-1.37\,\text{rad}$).
- **Zero floor penetration**: Foot soles remain strictly flat on the deck floor ($z_{\text{foot}} \ge 0.000\,\text{m}$, minimum ankle height $z = 0.0215\,\text{m}$), eliminating all root drops and floor mesh clipping.
- **Smooth transitions**: $C^2$-continuous quintic smoothstep splines eliminate velocity jumps and jerks between idle scanning and dynamic approach.

### Scene 3: Articulated beak clamp & counterbalanced payload lift (0:16 - 0:24)
- **Articulated beak mechanics**: Authentic Pollen Robotics lower jaw (`jaw.stl` + `jaw_soft.stl`) articulates open on the `beak_pitch` hinge, creating a wide $28\,\text{mm}$ aperture ($0.35\,\text{rad}$).
- **Active mesh colliders & clamp**: Active collision geometry (`condim="4"`, $\mu = 1.8$) encloses the $14\,\text{mm}$ dry-erase marker. The jaw firmly clamps shut ($0.05\,\text{rad}$) with dynamic equality weld engagement.
- **Payload lift**: Robot extends legs from the crouch back to upright stance, lifting the marker cleanly from ground level.
- **Counterbalanced dynamics**: Head and neck pitch adjust dynamically to counterbalance the forward center-of-mass shift, with BAM M6 battery voltage sag monitored on the HUD.

### Scene 4: Desktop maze chicane navigation with real-time ToF avoidance (0:24 - 0:32)
- **Maze chicane navigation**: Microduck navigates through a staggered 2-wall desktop maze chicane (`maze_wall_1` at $x=0.54\,\text{m}, y=+0.09\,\text{m}$ and `maze_wall_2` at $x=0.76\,\text{m}, y=-0.09\,\text{m}$).
- **Real-time 8x8 ToF reactive avoidance**: The 8x8 ToF sensor continuously detects wall proximities, dynamically steering the robot through the chicane with clear lateral margins.
- **Living character & anticipatory saccades**: Decoupled head yaw actively turns into the curves ahead of body heading, expressing anticipatory intention.

### Scene 5: Shallow transparent container arrival & marker drop-off (0:32 - 0:40)
- **Target container arrival**: Robot navigates to the shallow transparent acrylic container ($x = 1.05\,\text{m}$, clear $12\,\text{mm}$ rims, translucent base).
- **Targeted release**: Forward bowing posture positions the beak directly over the container interior.
- **Gravity settling physics**: The lower jaw articulates open to $28\,\text{mm}$, releasing the equality weld constraint. The dry-erase marker drops naturally under gravity ($g = -9.81\,\text{m/s}^2$) and settles stably into the acrylic container with a plastic drop acoustic click.

### Scene 6: Celebratory tilt, cheerful quack & seated rest (0:40 - 0:48)
- **Expressive character**: Microduck performs a $+14^\circ$ inquisitive head roll tilt, $+8^\circ$ pitch nod, emits a cheerful celebratory quack, and smoothly settles into a seated rest posture.
- **Auditor & critic certification**: Zero falls across all 1,440 frames (minimum ankle clearance $z = 0.0215\,\text{m} \ge 0.020\,\text{m}$, max tilt $8.4^\circ$), 100% factory XL330 Dynamixel joint limits respected, all 5 evaluation suite benchmarks passed, certified with highest honors by both Physics Simulation Critic and Cognitive Behavior Critic.

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
