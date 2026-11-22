"""Tests for CUDA RL Testbench and GPU policy evaluation."""

import pytest
import torch
from microduck_brain.sim.cuda_testbench import MicroduckCUDATestbench, TorchPolicyFromONNX


def test_torch_policy_from_onnx_ort_parity():
    """Verify TorchPolicyFromONNX matches ONNX Runtime (including obs_normalizer) to < 1e-5."""
    policy = TorchPolicyFromONNX("models/alpha_walking.onnx", device=torch.device("cpu"))
    max_abs = policy.verify_ort_parity(num_samples=32, seed=7)
    assert max_abs <= 1e-5, f"Expected <= 1e-5 ORT parity, got {max_abs:.3e}"


def test_cuda_testbench_execution():
    """Verify testbench runs real ExactBamPlant MuJoCo rollouts with valid metrics."""
    testbench = MicroduckCUDATestbench(policy_path="models/alpha_walking.onnx", device="cpu")

    # Fast test run with 16 parallel envs, 50 steps
    res = testbench.run_stress_test(
        num_parallel_envs=16,
        num_steps=50,
        lateral_push_max_mps=0.20,
    )

    assert res.total_samples == 16 * 50
    assert res.elapsed_seconds > 0.0
    assert res.throughput_samples_per_sec > 100.0
    assert res.ort_parity_max_abs <= 1e-5
    assert res.zero_fall_rate >= 0.99
    assert res.max_tilt_deg < 45.0
    assert res.pass_certification is True
