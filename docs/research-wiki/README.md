# Microduck Research Wiki — North Star & Multi-Agent Operating Protocol

> **Single source of truth** shared across both repositories:
> - **Skill Factory:** `microduck-reactor` (`D:\Codex_Archive\2026-09-04\files-pasted-by-the-user-use`)
> - **Embodied Brain & Deployment:** `microduck-robot-brain` (`c:\Users\ericr\Documents\antigravity\radiant-davinci`)

---

## Over-Arching Goal (1C — Phased Sim-First → Physical CM4)

Build an **expressive, resilient, autonomous Microduck** (12 cm bipedal duck, 14 Dynamixel XL330 servos, Raspberry Pi CM4) in two disciplined phases:

### Phase 1 (Current — Simulation)
Deliver **both** halves to real, verifiable engineering completion on Exact XL330 M6 BAM physics (`MuJoCo` + `mjlab`), with **zero sim-only cheats** that break hardware transfer:
1. **Verified Skill Library (Standing + Dynamic Stepping):**
   - Standing expressive skills (`bow`, `bow_walk_entry`, `wave`/`byebye`, `call`) and dynamic stepping/balance skills (`kick`, `jazzwalk`, `karate_kicks`) compiled to self-contained `61 -> 14` `official61` ONNX policies.
   - Every promoted skill must pass:
     - **Kinematic motion fidelity** (`evaluate_skill_fidelity()` — proves the duck actually performs the target gesture, never stand-in-place),
     - **Target 1 & Target 2** on P1 (`>= 90%` success across nominal, P1, P1×1.5, and push-or-get-up),
     - **Target 5 hardware proxies (`v2_calibrated`)** (`>= 90%` pass on non-push trials: torque `<= 1%` above 90% stall, slip `<= 0.02 m/s`, penetration `<= 3 mm`, sole force `<= 3.0 BW`, 20 ms debounced lift-offs, `1e-4 rad^2` `>3 Hz` tilt floor),
     - **Seamless walk → skill → walk chaining** and **autonomous fall → get-up → walk recovery** through `MicroduckLocomotionEngine`.
2. **True Closed-Loop Autonomous Brain (Zero Teleport / Zero Ground-Truth Cheats):**
   - Natural-language/voice intent parsing → Reactive Behavior Trees → real closed-loop perception (RGB camera detector from `render_rgb()`, raycast 8×8 ToF depth, IMU + battery brownout guardrails) driving walking, obstacle avoidance, visual approach, expressive skills, and fall recovery on `ExactBamPlant` / `use_exact_bam=True`.

### Phase 2 (Next — Physical CM4 Sim-to-Real)
Deploy the exact Phase 1 ONNX skills and executive brain onto the physical Raspberry Pi CM4 (`PollenRobotdClient` + `PollenTofClient`, Core 3 RT isolation), verify real-hardware telemetry against simulation predictions, and log sim-to-real discrepancies back into this wiki.

---

## Our "Lean" Verifier (Ground Truth Gate)

Unlike formal math, robotics has no symbolic proof checker. Our automated verifier stack plays the role of Lean:

1. **Reactor Gate (CPU, ~20 s):**
   ```bash
   wsl -d Ubuntu -- bash -lc "cd /mnt/d/Codex_Archive/2026-09-04/files-pasted-by-the-user-use && \
     CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable PYTHONPATH=.:src \
     /home/ericr/.venvs/microduck-reactor-gate4/bin/python -B scripts/quick_skill_gate.py \
     --onnx <policy.onnx> --motion <motion.npz> --target <target.npz> \
     --contract <residual61|official61> --p1 10 --p1x15 5 --pushes-per-slot 2"
   ```
2. **Brain Integration & Motion-Fidelity Gate (Windows, ~13 s):**
   ```powershell
   C:\Python313\python.exe -m pytest tests/ -q
   C:\Python313\python.exe scripts/run_brain_eval_suite.py
   ```

**Hard Wiki Rule:** *No claim, recipe, or policy promotion may be recorded as "verified" in this wiki without the exact command, artifact path, and output numbers from the verifier gates above.*

---

## Multi-Agent Lab Topology (3 Execution Labs + 1 Synthesis Coordinator)

Designed for a **single shared RTX 5090** + parallel CPU cores:

| Lab | Compute Domain | Owns |
| --- | --- | --- |
| **1. Motion & Feasibility Lab** | CPU-parallel | Mocap retargeting, morphological CoM/torque feasibility checks, Stage-1 realized motion extraction (`extract_stage1_realized_motion`) before GPU spend. |
| **2. RL Skill Factory Lab** | **Exclusive owner of the serialized RTX 5090 queue** | Runs one GPU training job at a time (`standing_v2`, `stepping_stage1`, `stepping_stage2`). Always checks `nvidia-smi` first (`--num-envs 2048` when GPU is shared, `4096` when free). |
| **3. Verifier & Brain Integration Lab** | CPU-parallel | Runs `quick_skill_gate.py`, compiles passing checkpoints into `microduck-robot-brain/models/`, updates `manifest.json`, and builds true closed-loop sim perception (RGB detector + raycast 8×8 ToF + real BT missions). |
| **4. Synthesis Coordinator** | Light CPU / Wiki | Runs periodic **Virtual Research Conferences** after each completed GPU run or milestone batch; updates [LEADERBOARD.md](LEADERBOARD.md), [FINDINGS_AND_DEAD_ENDS.md](FINDINGS_AND_DEAD_ENDS.md), [EXPERIMENT_QUEUE.md](EXPERIMENT_QUEUE.md), and [CONFERENCE_LOG.md](CONFERENCE_LOG.md). |

### Hard Anti-Spiral Guardrails
1. **No meta-audit bureaucracy:** Never run governance-only or SHA-paperwork-only rounds (preventing a repeat of the historical 63-round critic spiral). Every conference must either **promote a verified artifact**, **prune a dead-end**, or **enqueue a concrete physics/code experiment**.
2. **One GPU job at a time:** Only the RL Skill Factory Lab launches CUDA workloads.
3. **Preserve workspace safety:** Never run destructive git commands (`git reset --hard`, `git clean -fd`, `git checkout .`, `git push --force`) and never touch untracked non-Microduck directories in `radiant-davinci`.

---

## Wiki Index

- [LEADERBOARD.md](LEADERBOARD.md) — Current verified skill metrics, promotion status, and Brain test status.
- [EXPERIMENT_QUEUE.md](EXPERIMENT_QUEUE.md) — Ranked GPU training queue and parallel CPU Brain/Motion workstreams.
- [FINDINGS_AND_DEAD_ENDS.md](FINDINGS_AND_DEAD_ENDS.md) — Verified technical discoveries and pruned dead-ends.
- [CONFERENCE_LOG.md](CONFERENCE_LOG.md) — Virtual Research Conference protocol and session digests.
