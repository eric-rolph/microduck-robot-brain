"""Compile and package ReActor 61-D -> 14-D ONNX skills into models/ and verify parity.

Packages the following self-contained ``61 -> 14`` ``official61`` ONNX policies:
  1. ``bow.onnx``:
     Compiled from the selected iteration-2500 ``residual61`` checkpoint of ``bow_normal``
     (``docs/evidence/2026-10-18-checkpoint-selection-and-parity/selection/selected/policy.onnx``)
     and ``work/gpu-lane-residual/bow_normal/train/motion.npz`` (479 frames, 478 controls).
  2. ``bow_walk_entry.onnx``:
     Compiled from ``work/bow-r4-entry-ppo-002/export-robust_v3/policy.onnx``
     (229 frames, ``microduck_finite_phase_v1``, walk-entry bow policy).
  3. ``kick.onnx``:
     Compiled from the selected C@3500 ``official61`` checkpoint of ``M_ShortMove13``
     (``work/kick-acceptance-001/onnx-c/model_3500/policy.onnx`` + ``work/kick-tracking-b-001/motion.npz``,
     649 frames, 648 controls).
  4. ``wave.onnx`` (and alias ``byebye.onnx``):
     Compiled from ``work/gpu-lane/byebye_happy/train/selected/policy.onnx`` + ``motion.npz``
     (615 frames, 614 controls, 29/29 Target-1 completions with zero falls).
  5. ``call.onnx``:
     Compiled from ``work/gpu-lane/call_normal/train/selected/policy.onnx`` + ``motion.npz``
     (534 frames, 533 controls, 1.00 signature event recall, 0.072 rad RMSE).
  6. ``karate_kicks.onnx``:
     Compiled from ``work/gpu-lane/g1_karate_kicks13/train/selected/policy.onnx`` + ``motion.npz``
     (699 frames, 698 controls, 61->20 training-wheels policy sliced to strict 61->14).
  7. ``jazzwalk.onnx``:
     Compiled from ``work/gpu-lane/g1_jazzwalk/train/selected/policy.onnx`` + ``motion.npz``
     (781 frames, 780 controls, 61->20 training-wheels policy sliced to strict 61->14, 29/29 Target-1).
  8. ``BEST_alpha_walking.onnx`` & ``BEST_alpha_stand.onnx``:
     Official pinned Pollen locomotion and get-up recovery policies (61 -> 14).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microduck_brain.reactor_onnx_compiler import (  # noqa: E402
    OFFICIAL_DEFAULT_POSE,
    PHASE_SCHEMA_FINITE_V1,
    PHASE_SCHEMA_FOURIER_V1,
    compile_self_contained_pollen61_onnx,
    compute_phase_command,
    evaluate_skill_fidelity,
    read_reactor_onnx_metadata,
    sha256_file,
)

DEFAULT_REACTOR_REPO = Path(
    os.environ.get("MICRODUCK_REACTOR_REPO", r"D:\Codex_Archive\2026-09-04\files-pasted-by-the-user-use")
)


def verify_residual_wrapper_parity(
    raw_residual_onnx: Path,
    compiled_official_onnx: Path,
    motion_npz: Path,
    *,
    seed: int = 42,
) -> dict[str, float | int | bool]:
    """Verify closed-loop multi-step trajectory equivalence between residual61 and compiled official61."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess_res = ort.InferenceSession(str(raw_residual_onnx), opts, providers=["CPUExecutionProvider"])
    sess_off = ort.InferenceSession(str(compiled_official_onnx), opts, providers=["CPUExecutionProvider"])

    with np.load(motion_npz) as raw:
        frames = int(len(raw["source_frame_index"]))
        commands = np.asarray(raw["commands"], dtype=np.float64)

    T = len(commands)
    rng = np.random.default_rng(seed)
    last_raw_res = np.zeros(14, dtype=np.float32)
    last_raw_off = np.zeros(14, dtype=np.float32)
    max_target_diff = 0.0
    max_handover_diff = 0.0

    for k in range(T):
        sensors_34 = rng.normal(0.0, 0.2, size=(34,)).astype(np.float32)
        phase_13 = compute_phase_command(k, frames, phase_schema=PHASE_SCHEMA_FOURIER_V1)

        obs_res = np.concatenate([sensors_34, last_raw_res, phase_13]).astype(np.float32)[None, :]
        obs_off = np.concatenate([sensors_34, last_raw_off, phase_13]).astype(np.float32)[None, :]

        act_res = np.asarray(sess_res.run(None, {"obs": obs_res})[0][0], dtype=np.float64)[:14]
        act_off = np.asarray(sess_off.run(None, {"obs": obs_off})[0][0], dtype=np.float64)[:14]

        target_res = commands[min(k, T - 1)] + act_res
        target_off = OFFICIAL_DEFAULT_POSE.astype(np.float64) + act_off
        diff = float(np.max(np.abs(target_res - target_off)))
        max_target_diff = max(max_target_diff, diff)

        last_raw_res = act_res.astype(np.float32)
        last_raw_off = act_off.astype(np.float32)

    handover_res = (commands[T - 1] + last_raw_res.astype(np.float64)) - OFFICIAL_DEFAULT_POSE.astype(np.float64)
    handover_off = last_raw_off.astype(np.float64)
    max_handover_diff = float(np.max(np.abs(handover_res - handover_off)))

    return {
        "steps_tested": T,
        "max_target_abs_diff": max_target_diff,
        "max_handover_abs_diff": max_handover_diff,
        "parity_pass": bool(max_target_diff <= 1e-6 and max_handover_diff <= 1e-6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactor-repo", type=Path, default=DEFAULT_REACTOR_REPO)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--strict-fidelity",
        action="store_true",
        help="Fail if any packaged skill fails kinematic motion-fidelity floors on ExactBamPlant.",
    )
    args = parser.parse_args(argv)

    reactor_repo = args.reactor_repo.resolve()
    models_dir = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "skill_name": "bow",
            "output_file": "bow.onnx",
            "source_onnx": reactor_repo
            / "docs/evidence/2026-10-18-checkpoint-selection-and-parity/selected2500/policy.onnx",
            "motion_npz": reactor_repo / "work/gpu-lane-residual/bow_normal/train/motion.npz",
            "source_contract": "residual61",
            "phase_schema": PHASE_SCHEMA_FOURIER_V1,
            "frames": None,
        },
        {
            "skill_name": "bow_walk_entry",
            "output_file": "bow_walk_entry.onnx",
            "source_onnx": reactor_repo / "work/bow-r4-entry-ppo-002/export-robust_v3/policy.onnx",
            "motion_npz": None,
            "source_contract": "official61",
            "phase_schema": PHASE_SCHEMA_FINITE_V1,
            "frames": 229,
        },
        {
            "skill_name": "kick",
            "output_file": "kick.onnx",
            "source_onnx": reactor_repo / "work/kick-acceptance-001/onnx-c/model_3500/policy.onnx",
            "motion_npz": reactor_repo / "work/kick-tracking-b-001/motion.npz",
            "source_contract": "official61",
            "phase_schema": PHASE_SCHEMA_FOURIER_V1,
            "frames": None,
        },
        {
            "skill_name": "wave",
            "output_file": "wave.onnx",
            "source_onnx": reactor_repo / "work/gpu-lane/byebye_happy/train/selected/policy.onnx",
            "motion_npz": reactor_repo / "work/gpu-lane/byebye_happy/train/motion.npz",
            "source_contract": "official61",
            "phase_schema": PHASE_SCHEMA_FOURIER_V1,
            "frames": None,
        },
        {
            "skill_name": "call",
            "output_file": "call.onnx",
            "source_onnx": reactor_repo / "work/gpu-lane/call_normal/train/selected/policy.onnx",
            "motion_npz": reactor_repo / "work/gpu-lane/call_normal/train/motion.npz",
            "source_contract": "official61",
            "phase_schema": PHASE_SCHEMA_FOURIER_V1,
            "frames": None,
        },
        {
            "skill_name": "karate_kicks",
            "output_file": "karate_kicks.onnx",
            "source_onnx": reactor_repo / "work/gpu-lane/g1_karate_kicks13/train/selected/policy.onnx",
            "motion_npz": reactor_repo / "work/gpu-lane/g1_karate_kicks13/train/motion.npz",
            "source_contract": "official61",
            "phase_schema": PHASE_SCHEMA_FOURIER_V1,
            "frames": None,
        },
        {
            "skill_name": "jazzwalk",
            "output_file": "jazzwalk.onnx",
            "source_onnx": reactor_repo / "work/gpu-lane/g1_jazzwalk/train/selected/policy.onnx",
            "motion_npz": reactor_repo / "work/gpu-lane/g1_jazzwalk/train/motion.npz",
            "source_contract": "official61",
            "phase_schema": PHASE_SCHEMA_FOURIER_V1,
            "frames": None,
        },
    ]

    report: dict[str, Any] = {"skills": {}, "locomotion": {}}
    for spec in specs:
        out_path = models_dir / spec["output_file"]
        meta = compile_self_contained_pollen61_onnx(
            source_onnx_path=spec["source_onnx"],
            output_onnx_path=out_path,
            motion_npz_path=spec["motion_npz"],
            source_contract=spec["source_contract"],
            phase_schema=spec["phase_schema"],
            frames=spec["frames"],
            skill_name=spec["skill_name"],
        )
        if spec["source_contract"] == "residual61":
            parity = verify_residual_wrapper_parity(
                spec["source_onnx"], out_path, spec["motion_npz"]
            )
            meta["residual_to_official_parity"] = parity
            if not parity["parity_pass"]:
                raise RuntimeError(f"Residual wrapper parity failed for {spec['skill_name']}: {parity}")

        # Verify metadata round-trip and real kinematic motion fidelity on ExactBamPlant
        readback = read_reactor_onnx_metadata(out_path)
        assert readback["frames"] == meta["frames"]
        assert readback["phase_schema"] == meta["phase_schema"]
        fidelity = evaluate_skill_fidelity(out_path, skill_name=spec["skill_name"])
        meta["kinematic_fidelity"] = fidelity
        meta["status"] = (
            "provisional_verified_motion" if fidelity["fidelity_pass"] else "experimental_failing_fidelity"
        )
        if args.strict_fidelity and not fidelity["fidelity_pass"]:
            raise RuntimeError(
                f"Skill {spec['skill_name']!r} failed motion-fidelity gate: {fidelity['failures']}"
            )
        report["skills"][spec["skill_name"]] = meta

    # Copy wave.onnx to byebye.onnx as alias
    shutil.copyfile(models_dir / "wave.onnx", models_dir / "byebye.onnx")

    # Ensure official locomotion and stand recovery policies exist in models/
    loco_base = reactor_repo / "work/recovery-official-source-001/app/public/policies"
    for name in ("BEST_alpha_walking.onnx", "BEST_alpha_stand.onnx"):
        src = loco_base / name
        dst = models_dir / name
        if src.exists():
            shutil.copyfile(src, dst)
            report["locomotion"][name] = {
                "path": str(dst),
                "sha256": sha256_file(dst),
            }
    # Keep alpha_walking.onnx synced with BEST_alpha_walking.onnx
    if (models_dir / "BEST_alpha_walking.onnx").exists():
        shutil.copyfile(models_dir / "BEST_alpha_walking.onnx", models_dir / "alpha_walking.onnx")

    report_path = models_dir / "reactor_compiled_skills.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
