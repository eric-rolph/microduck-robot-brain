"""Tests for CUDA RL Testbench and GPU policy evaluation."""

import pytest
import torch
from microduck_brain.sim.cuda_testbench import MicroduckCUDATestbench


def test_cuda_testbench_execution():
    """Verify testbench runs batched evaluations with valid metrics."""
    testbench = MicroduckCUDATestbench(policy_path="models/alpha_walking.onnx")

    # Fast test run with 16 parallel envs, 50 steps
    res = testbench.run_stress_test(
        num_parallel_envs=16,
        num_steps=50,
        lateral_push_max_mps=0.20,
    )

    assert res.total_samples == 16 * 50
    assert res.elapsed_seconds > 0.0
    assert res.throughput_samples_per_sec > 100.0
    assert res.zero_fall_rate >= 0.99
    assert res.max_tilt_deg < 45.0
    assert res.pass_certification is True
