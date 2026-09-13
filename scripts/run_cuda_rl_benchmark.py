"""Executable script to benchmark Microduck RL policy on NVIDIA GeForce RTX 5090."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
import torch

from microduck_brain.sim.cuda_testbench import MicroduckCUDATestbench


def main() -> None:
    print("=" * 65)
    print(" MICRODUCK ROBOT BRAIN // RTX 5090 CUDA RL TESTBENCH")
    print("=" * 65)

    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available. Running on CPU fallback.")
    else:
        dev_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Active Compute Device: {dev_name} ({vram_gb:.1f} GB VRAM, CUDA {torch.version.cuda})")

    policy_path = Path("models/alpha_walking.onnx")
    if not policy_path.exists():
        print(f"Error: Policy file not found: {policy_path}")
        sys.exit(1)

    testbench = MicroduckCUDATestbench(policy_path=policy_path)
    print("\nRunning batched GPU stress test: 256 parallel environments, 500 control steps...")
    result = testbench.run_stress_test(
        num_parallel_envs=256,
        num_steps=500,
        lateral_push_max_mps=0.45,
        obs_noise_std=0.02,
    )

    print("\n" + "=" * 65)
    print(" BENCHMARK RESULTS")
    print("=" * 65)
    print(f"  Device:                  {result.device_name}")
    print(f"  Parallel Environments:   {result.num_parallel_envs}")
    print(f"  Control Steps / Env:     {result.num_steps}")
    print(f"  Total Simulated Steps:   {result.total_samples:,}")
    print(f"  Elapsed Wall Time:       {result.elapsed_seconds:.3f} s")
    print(f"  Throughput:              {result.throughput_samples_per_sec:,.0f} steps/second")
    print(f"  Zero Fall Survival Rate: {result.zero_fall_rate * 100:.1f} %")
    print(f"  Mean Posture Tilt:       {result.mean_tilt_deg:.2f}°")
    print(f"  Max Posture Tilt:        {result.max_tilt_deg:.2f}°")
    print(f"  Certification Status:    {'CERTIFIED (PASS)' if result.pass_certification else 'FAILED'}")
    print("=" * 65)

    out_json = Path("output/cuda_rl_benchmark.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result.__dict__, f, indent=2)
    print(f"Saved benchmark report to {out_json}")


if __name__ == "__main__":
    main()
