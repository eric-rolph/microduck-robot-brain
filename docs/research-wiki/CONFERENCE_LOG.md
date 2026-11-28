# Virtual Research Conference Log

## Conference Protocol (Run After Each GPU Job or Milestone Batch)

Every conference entry MUST stay concise (< 40 lines) and answer **four concrete questions**:
1. **Verified Results Since Last Conference:** Exact artifact paths + `quick_skill_gate.py` / `pytest` / `run_brain_eval_suite.py` numbers.
2. **Leaderboard Changes:** Skills or Brain capabilities promoted, demoted, or updated in [LEADERBOARD.md](LEADERBOARD.md).
3. **Dead-Ends Pruned / New Insights:** Added to [FINDINGS_AND_DEAD_ENDS.md](FINDINGS_AND_DEAD_ENDS.md).
4. **Next Assignment (1 GPU Job + Parallel CPU Tasks):** Exact next owner and command from [EXPERIMENT_QUEUE.md](EXPERIMENT_QUEUE.md).

*Forbidden:* Adding meta-audit checklists or process bureaucracy without an empirical robotics finding.

---

## Conference #0 — Baseline Recovery & Wiki Kickoff (2026-11-24)

### 1. Verified Results
- **`microduck-robot-brain` (Track 1 fixes):**
  - `pytest tests/ -q`: **95/95 passed** in `12.57s` (down from `28.17s` via `ExactBamPlant` model caching).
  - `scripts/run_brain_eval_suite.py`: **5/5 passed** (max tilt `7.7°` on Fetch + 0.20 m/s push).
  - `evaluate_skill_fidelity()` confirmed `bow` (`20.7°` pitch), `bow_walk_entry` (`14.2°` pitch), and `kick` (`89.2 mm` foot lift) pass; `wave`, `call`, `karate_kicks`, and `jazzwalk` are honestly quarantined as `experimental_failing_fidelity`.
- **`microduck-reactor` (Track 2 fixes):**
  - Core + Track 2 `pytest` (WSL2 CPU): **134/134 passed** in `38.10s`.
  - `scripts/quick_skill_gate.py` on `bow_normal` (`selected2500/policy.onnx`, `residual61`, 22 eps, `18.5s`):
    - Nominal `1/1`, P1 `9/10` (T1 `10/10`, T2 `9/10`, RMSE `0.057 rad`), P1×1.5 `4/5`, Push `6/6` (100%).
    - Hardware Target 5: `v1_strict` `15/22` → **`v2_calibrated` `20/22` (`16/16 = 100%` non-push)**.
  - Verified `extract_stage1_realized_motion()` on `g1_jazzwalk` (60-frame smoke slice, `stiff_frictionloss=True`, 0 falls) loads and steps cleanly in `Residual61Env`.

### 2. Leaderboard & Goal Lock-In
- User confirmed **Goal 1C (Sim-first Skill Factory + Autonomous Closed-Loop Brain → Phase 2 Physical CM4)** and confirmed we are currently in **Phase 1 (Sim)**.
- Seeded [LEADERBOARD.md](LEADERBOARD.md), [EXPERIMENT_QUEUE.md](EXPERIMENT_QUEUE.md), and [FINDINGS_AND_DEAD_ENDS.md](FINDINGS_AND_DEAD_ENDS.md).

### 3. Dead-Ends Pruned
- Quarantined Round-1 `absolute` standing policies and single-stage wheels-fade stepping policies.
- Confirmed `chosen_actuator` XML damping must not be overwritten in `position_limited` mode.
- Confirmed `nvidia-smi` shows RTX 5090 currently shared (`13.1 / 32.6 GiB` VRAM in use) → default to `--num-envs 2048` for shared GPU training.

### 4. Next Assignment (Ready for Conference #1)
- **RL Skill Factory Lab (GPU #1):** Train `bow_normal` with `--recipe standing_v2 --num-envs 2048` → gate with `quick_skill_gate.py`.
- **Motion & Feasibility Lab (CPU-1):** Run full-clip CPU Stage-1 PD realizability audit on `g1_jazzwalk` and `g1_karate_kicks13`.
- **Verifier & Brain Lab (CPU-2 + CPU-4):** Baseline `kick` on `quick_skill_gate.py`, and design/implement real RGB-only `HeadCamera` detection + raycast 8×8 ToF sensor on `ExactBamPlant`.
