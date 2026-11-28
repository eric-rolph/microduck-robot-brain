# Microduck Verified Leaderboard

Updated: **2026-11-24** (Conference #0 — Post Track 1 & Track 2 Recovery Baseline)

---

## 1. ReActor Skill Policy Leaderboard (`microduck-robot-brain/models/`)

Promotion requires passing **both**:
1. `evaluate_skill_fidelity()` kinematic motion floor in `microduck-robot-brain`, AND
2. `quick_skill_gate.py` (`v2_calibrated`) `>= 90%` P1 success & `>= 90%` non-push hardware pass in `microduck-reactor`.

| Skill | Status in `manifest.json` | Source Recipe | Fidelity Gate (`ExactBamPlant`) | Key Measured Motion | Reactor Gate (`quick_skill_gate.py`, 22 eps) | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| `bow` | `provisional_verified_motion` | `residual61` iter 2500 (`selected2500`) | **PASS** | Forward pitch `20.7°`, leg ptp `0.566 rad`, mean spd `0.056 rad/s` | P1 `9/10` (T1 `10/10`, T2 `9/10`, RMSE `0.057 rad`), P1×1.5 `4/5`, Push `6/6`, HW v1 `15/22` → **HW v2 `20/22` (`16/16` non-push!)** | Retrain with `standing_v2` (`event_weight=1.0`, `action_rate_final=-0.12`) to push P1 `9/10` → `10/10` (`sway_right` recall). |
| `bow_walk_entry` | `provisional_verified_motion` | CPU/PPO walk-entry `bow-r4-entry-ppo-002` | **PASS** | Forward pitch `14.2°`, head ptp `0.643 rad`, mean spd `0.073 rad/s` | Not yet run through `quick_skill_gate.py` | Baseline in `quick_skill_gate.py`; keep as walk-entry specialist. |
| `kick` | `provisional_verified_motion` | `official61` (`kick-acceptance-001`) | **PASS** | Right foot lift `89.2 mm`, pitch `[-29.7°, +8.6°]`, leg ptp `1.683 rad`, mean spd `0.400 rad/s` | Legacy acceptance track passed T1/T2; run `quick_skill_gate.py` for v2 HW score | Run `quick_skill_gate.py` to log exact v1 vs v2 HW numbers. |
| `wave` (`byebye`) | `experimental_failing_fidelity` | Round-1 `absolute` (`gpu-lane/byebye_happy`) | **FAIL** (rejected) | Head ptp `0.381 rad` (`< 0.50`), mean spd `0.055 rad/s` (`< 0.06`) | `0/40` P1 in Round 1 | Retrain with `--recipe standing_v2` (GPU Queue #2). |
| `call` | `experimental_failing_fidelity` | Round-1 `absolute` (`gpu-lane/call_normal`) | **FAIL** (rejected) | Head ptp `0.597 rad` (`< 0.65`), mean spd `0.066 rad/s` (`< 0.075`) | `0/40` P1 in Round 1 | Retrain with `--recipe standing_v2` (GPU Queue #3). |
| `jazzwalk` | `experimental_failing_fidelity` | Round-1 single-stage wheels fade | **FAIL** (rejected) | Left foot `3.4 mm`, right foot `11.0 mm` (`< 25 mm`) — stands in place | `0/40` P1 in Round 1 | Run genuine 2-stage ReActor (`run_two_stage_reactor.py`, GPU Queue #4). |
| `karate_kicks` | `experimental_failing_fidelity` | Round-1 single-stage wheels fade | **FAIL** (rejected) | Left foot `2.4 mm`, right foot `13.8 mm` (`< 40 mm`) — stands in place | `0/40` P1 in Round 1 | Run genuine 2-stage ReActor (`run_two_stage_reactor.py`, GPU Queue #5). |

---

## 2. Executive Brain & Simulation Verifier Status

| Subsystem | Current Verified Score | Notes / Open Gap |
| --- | --- | --- |
| `microduck-robot-brain` unit & E2E tests | **95/95 PASSED** (`12.57s`) | Includes ORT parity `<= 1e-5`, positive/negative skill fidelity gates, `ReactiveSelectorNode` physical bridge, `use_exact_bam=True`. |
| `scripts/run_brain_eval_suite.py` | **5/5 PASSED** | Fetch+push (max tilt `7.7°`), brownout hysteresis, fallen safety abort, ambient balance gating, stability envelope. |
| `microduck-reactor` core & Track 2 tests | **134/134 PASSED** (`38.10s`) | Includes `v1_strict` preservation, `v2_calibrated` debounce & tilt floor, `stiff_frictionloss`, `standing_v2` event windows, Stage-1 realized motion extraction. |
| Autonomous Sim Perception (Vision + ToF) | **GAP — Phase 1 Priority** | `HeadCamera.detect_objects()` still reads MuJoCo geom segmentation (`seg_renderer`), and `render_demo_video.py` uses scripted timers + weld-teleport grasp. Needs real RGB color/blob detector + MuJoCo raycast 8×8 ToF + closed-loop BT mission runner on `ExactBamPlant`. |
| Physical CM4 Deployment | **Phase 2 (Deferred)** | Core 3 `systemd` affinity + 14-DOF `duck_shm.hpp` + `PhysicalMicroduckController` fixed and unit-tested; awaits physical duck availability. |
