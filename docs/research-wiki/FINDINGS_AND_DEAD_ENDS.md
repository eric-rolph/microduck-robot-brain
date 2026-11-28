# Verified Findings & Pruned Dead-Ends

> Institutional memory extracted from empirical failures and fixes across both repos. Read this before proposing any new reward, evaluation rule, or architecture change.

---

## Verified Technical Findings

1. **Action-rate over-smoothing destroys subtle events in late training:**
   - In `microduck-reactor`, `action_rate_final=-0.4` with `joint_pos_weight=0.0` caused `bow_normal` (`residual61`) to peak at iter 2500 (`26/29` task, `18/29` joint) and collapse to `2/29` by iter 2999 by smoothing away the `sway_right` roll excursion.
   - **Fix implemented:** `RECIPES["standing_v2"]` sets `action_rate_final=-0.12`, `joint_pos_weight=1.0`, `joint_pos_std=0.15`, and `event_weight=1.0` (`motion_event_joint_pos_exp` inside signature-event windows).

2. **`v1_strict` Target 5 hardware proxies reject clean closed-loop policies on solver flicker and microscopic jitter:**
   - Empirical 22-episode CPU gate on `bow_normal` (`selected2500/policy.onnx`):
     - `v1_strict`: `15/22` pass (`4` fails on `tilt_energy`, `3` fails on `lift_off_count`), failing even nominal-looking P1/P1×1.5 rollouts on single 5 ms sole-unload flickers and `1e-5 rad^2` jitter against ultra-smooth open-loop references.
     - `v2_calibrated` (`min_unload_steps=4` = 20 ms debounce, `tilt_energy_abs_floor_rad2=1e-4` ~ 0.57° RMS): **`20/22` pass (`16/16 = 100%` on all non-push episodes)** while still rejecting real push-induced foot lift-offs and falls.

3. **Target 6 latency ceiling (`+6` extra controls = +120 ms) exceeds physical inverted-pendulum controllability:**
   - Pollen's own shipped `BEST_alpha_walking.onnx` fails starting at `+1` extra control step (+20 ms on top of base 20 ms delay). For a 12 cm bipedal robot ($\sqrt{L/g} \approx 110\text{ ms}$), calibrated Target 6 sweeps `+0..1` extra controls (`20–40 ms` total).

4. **CPU BAM `stiff_frictionloss` parity:**
   - Upstream `work/gate2_upstream_probe/scripts/infer_policy.py` and `microduck_brain/sim/exact_bam_plant.py` both already compile `solref_friction = (-5e4, -2e2)` and `solimp_friction = (0.99, 0.9999, 0.001, 0.5, 2.0)` matching `bam.mjlab.BamActuator`.
   - `BamPlant` now exposes `stiff_frictionloss: bool = True` explicitly without mutating `_MODEL_CACHE`.

5. **Position-actuator `chosen_actuator` damping vs Exact BAM torque mode in `env.py`:**
   - `robot_allcollisions_backlash.xml`'s `chosen_actuator` class encodes `damping="0.053"` (motor back-EMF `kt^2/R ≈ 0.04` + viscous `0.012`). Overwriting `model.dof_damping` with `BamM6Config.viscous_damping = 0.006` in position-actuator mode strips back-EMF damping and destabilizes walking (caused Task 1 tilt `43°` at step 99).
   - **Rule:** Preserve XML `dof_damping`/`dof_frictionloss` in `position_limited` mode; use `use_exact_bam=True` (or `ExactBamPlant`) when full XL330 M6 torque + Stribeck dynamics are needed.

6. **Stateful `SelectorNode` traps `FETCH` in infinite search:**
   - `SelectorNode` caches `current_idx`, so once `FetchSearch` returned `RUNNING`, seeing the ball never preempted search.
   - **Fix implemented:** Added `ReactiveSelectorNode` in `microduck_brain/behavior_tree.py` (re-evaluates from index 0 every tick and aborts lower-priority branches on preemption).

7. **ONNX `obs_normalizer` must never be skipped:**
   - `models/alpha_walking.onnx` embeds `Sub` (`_mean`) and `Div` (`_std`) nodes before the MLP `Gemm`/`Elu` layers. `TorchPolicyFromONNX` now executes the full topological ONNX graph and asserts `<= 1e-4` (actual `<= 1e-5`) ORT parity on initialization.

8. **Raspberry Pi CM4 core topology:**
   - CM4 has 4 cores (`0–3`), not 5+. All `systemd` RT isolation files now pin `robotd` to Core 3 and `microduck-brain` to Cores 1–2.

---

## Pruned Dead-Ends (Do NOT Repeat)

| Dead-End | Why It Failed | Replacement |
| --- | --- | --- |
| **Single-stage linear wheels fade on unadapted stepping retargets** (`wheels_fade_start=1000 -> off=2500`) | When root wrench hits zero against an infeasible kinematic stepping reference, `ee_body_pos` terminations spike and the policy collapses to standing in place (`jazzwalk` 3.4 mm lift, `karate_kicks` 13.8 mm lift, `0/40` P1). | **2-stage ReActor** (`scripts/run_two_stage_reactor.py`): keep wheels active in Stage 1 → extract physically realized `stage1_realized_motion.npz` → train force-free `residual61` in Stage 2. |
| **Round-1 `action_mode="absolute"` on standing clips (`byebye`, `call`)** | Combined with `action_rate_final=-0.4` and `joint_pos_weight=0.0`, policies learned minimal-effort low-energy stand-in-place (`0/40` P1). | `action_mode="residual"` + `--recipe standing_v2`. |
| **Promotion gated only on `not plant.is_fallen()`** | Allowed standing-still `jazzwalk`/`karate_kicks` and under-moving `wave`/`call` to pass E2E tests and ship in `manifest.json`. | Mandatory `evaluate_skill_fidelity()` kinematic motion floors + `quick_skill_gate.py` v2 gate before promoting out of `experimental_failing_fidelity`. |
| **Audit-only governance rounds (historical rounds 43–63)** | Spent 20+ consecutive rounds hardening SHA/AST receipt parsers while zero new skills were trained or fixed. | Conferences must output verified physics metrics, promoted artifacts, or a ranked next experiment. |
| **Recursive PowerShell/WSL scans over `D:\Codex_Archive\2026-09-04` or `work/`** | `work/` has 1,539+ directories and `mdr-tracks/motion-variety/work` symlinks back to `work/`, causing timeouts. | Target exact subdirectories directly; run Windows `git` natively instead of WSL `git` on `/mnt/d/`. |
