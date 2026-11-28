# Microduck Experiment Queue

> **GPU Rule:** Only the **RL Skill Factory Lab** executes items in Section A, strictly **one job at a time**. Always check `wsl -d Ubuntu -- nvidia-smi` first: use `--num-envs 2048` when VRAM is shared (`> 4 GiB` in use), or `--num-envs 4096` when the RTX 5090 is idle.

---

## Section A — Serialized RTX 5090 GPU Queue (`microduck-reactor`)

### Priority 1: `bow_normal` with `standing_v2` (Target: `10/10` P1 + `100%` v2 HW)
- **Hypothesis:** `selected2500` already hits `9/10` P1 and `16/16` non-push `v2_calibrated` HW, missing only 1 P1 seed on the subtle `sway_right` signature event because `action_rate_final=-0.4` eroded it. Training with `--recipe standing_v2` (`action_rate_final=-0.12`, `joint_pos_weight=1.0`, `event_weight=1.0`) should lift P1 to `>= 95%` without late-iteration collapse.
- **Command (WSL2 Ubuntu):**
  ```bash
  cd /mnt/d/Codex_Archive/2026-09-04/files-pasted-by-the-user-use && \
  MUJOCO_GL=egl PYTHONPATH=.:src /home/ericr/.venvs/microduck-reactor-gate4/bin/python \
    scripts/train_clip_tracking.py \
    --job-dir work/gpu-lane-residual/bow_normal/job \
    --output work/gpu-lane-v2/bow_normal/train \
    --recipe standing_v2 --num-envs 2048 --iterations 3000
  ```
- **Gate:** Run `scripts/quick_skill_gate.py --contract residual61 --p1 20 --p1x15 10 --pushes-per-slot 5`. If P1 `>= 95%` and non-push `v2_calibrated` `>= 95%`, compile into `microduck-robot-brain/models/bow.onnx` and promote to `verified_v2`.

### Priority 2: `byebye_happy` (`wave`) with `standing_v2`
- **Hypothesis:** Round-1 `absolute` policy under-moved (`head_ptp=0.381 rad`, `0/40` P1). `residual61` + `standing_v2` anchors the nominal wave motion in the reference command and rewards event windows directly.
- **Command:**
  ```bash
  cd /mnt/d/Codex_Archive/2026-09-04/files-pasted-by-the-user-use && \
  MUJOCO_GL=egl PYTHONPATH=.:src /home/ericr/.venvs/microduck-reactor-gate4/bin/python \
    scripts/train_clip_tracking.py \
    --job-dir work/gpu-lane/byebye_happy/job \
    --output work/gpu-lane-v2/byebye_happy/train \
    --recipe standing_v2 --num-envs 2048 --iterations 3000
  ```
- **Gate:** `quick_skill_gate.py` + `evaluate_skill_fidelity()` (`max_head_ptp_rad >= 0.50`, `mean_joint_speed_rad_s >= 0.06`).

### Priority 3: `call_normal` (`call`) with `standing_v2`
- **Hypothesis:** Same as `byebye_happy` — switch from Round-1 `absolute` to `residual61` + `standing_v2`.
- **Command:**
  ```bash
  cd /mnt/d/Codex_Archive/2026-09-04/files-pasted-by-the-user-use && \
  MUJOCO_GL=egl PYTHONPATH=.:src /home/ericr/.venvs/microduck-reactor-gate4/bin/python \
    scripts/train_clip_tracking.py \
    --job-dir work/gpu-lane/call_normal/job \
    --output work/gpu-lane-v2/call_normal/train \
    --recipe standing_v2 --num-envs 2048 --iterations 3000
  ```

### Priority 4: `g1_jazzwalk` with Genuine 2-Stage ReActor (`run_two_stage_reactor.py`)
- **Hypothesis:** Keeping root wheels active in Stage 1 (`stepping_stage1`), capturing the physically realized contact-consistent rollout (`stage1_realized_motion.npz`), and training Stage 2 force-free (`stepping_stage2`) breaks the single-stage wheels-fade collapse (`3.4 mm` lift → real alternating footsteps `>= 25 mm`).
- **Command:**
  ```bash
  cd /mnt/d/Codex_Archive/2026-09-04/files-pasted-by-the-user-use && \
  MUJOCO_GL=egl PYTHONPATH=.:src /home/ericr/.venvs/microduck-reactor-gate4/bin/python \
    scripts/run_two_stage_reactor.py \
    --job-dir work/variety-002/g1_jazzwalk \
    --output work/reactor-2stage/g1_jazzwalk \
    --num-envs 2048 --stage1-iterations 2000 --stage2-iterations 3000
  ```

### Priority 5: `g1_karate_kicks13` with Genuine 2-Stage ReActor
- **Command:** Same as Priority 4 targeting `--job-dir work/gpu-lane/g1_karate_kicks13/job --output work/reactor-2stage/g1_karate_kicks13`.

---

## Section B — Parallel CPU Workstreams (Run Concurrently Without GPU)

### CPU-1 (Motion & Feasibility Lab): Pre-GPU Stage-1 PD Realizability Audit
- Run `extract_stage1_realized_motion(..., policy=None)` on `g1_jazzwalk` and `g1_karate_kicks13` (full clip length) on CPU `BamPlant(stiff_frictionloss=True)` to measure baseline `stage1_wrench_mean_bw`, foot clearances, and event recall **before** spending GPU hours.
- If baseline PD wrench exceeds `0.45 BW` mean, inspect whether the kinematic retarget needs slower tempo or narrower stride amplitude for Microduck's 12 cm leg geometry.

### CPU-2 (Verifier & Brain Lab): Real Closed-Loop Sim Perception (No Ground-Truth Cheats)
- Add an **RGB-only vision detector** mode in `microduck_brain/vision/head_camera.py` (HSV/contour or lightweight ONNX detector operating purely on `render_rgb()`, verified against `seg_renderer` ground-truth IoU in unit tests).
- Add a **MuJoCo raycast 8×8 ToF simulator** (`mj_ray` grid from the `tof` site) feeding `WorldState.update_from_tof_frame()` directly.

### CPU-3 (Verifier & Brain Lab): True Autonomous Closed-Loop Mission Benchmark on `ExactBamPlant`
- Build an autonomous sim mission harness (distinct from the presentation video script `render_demo_video.py`) where:
  - `IntentParser` → `ReactiveSelectorNode` Behavior Tree → RGB `HeadCamera` + raycast 8×8 ToF → `MicroduckLocomotionEngine` (`BEST_alpha_walking.onnx` + verified ReActor skills + `BEST_alpha_stand.onnx`) runs **100% closed-loop** on `ExactBamPlant` with P1 randomization and mid-mission pushes (no wall-clock scripting, no weld-teleporting).
- Add an automated pytest benchmark asserting zero falls, real obstacle avoidance, real visual approach, and clean walk → `bow`/`kick` → walk execution.

### CPU-4 (Verifier Lab): Baseline `kick` and `bow_walk_entry` on `quick_skill_gate.py`
- Run `scripts/quick_skill_gate.py` on `work/kick-acceptance-001/onnx-c/model_3500/policy.onnx` and `bow_walk_entry` to populate missing v1 vs v2 HW cells in [LEADERBOARD.md](LEADERBOARD.md).
