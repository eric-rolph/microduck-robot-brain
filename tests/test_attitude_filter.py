"""
Unit tests for AttitudeCommandFilter.
"""

import math
from microduck_brain.locomotion_engine import AttitudeCommandFilter


def test_attitude_filter_attenuation_at_high_speed():
    # max forward vel = 0.8
    filter_ = AttitudeCommandFilter(dt=0.02, max_rate_rad_s=0.5, max_forward_vel=0.8)

    # At full forward speed (vx >= 0.8), attitude commands are attenuated to 0
    roll, pitch = filter_.process(target_roll=0.15, target_pitch=0.15, target_vx=0.8)
    assert roll == 0.0
    assert pitch == 0.0


def test_attitude_filter_slew_rate_clamping():
    # dt=0.02, max_rate=0.5 rad/s -> max step per tick is 0.01 rad
    filter_ = AttitudeCommandFilter(dt=0.02, max_rate_rad_s=0.5, max_forward_vel=1.0)

    # Large sudden step of 0.2 rad
    roll, pitch = filter_.process(target_roll=0.20, target_pitch=0.0, target_vx=0.0)

    # Output should not exceed 0.01 on first tick
    assert math.isclose(roll, 0.01, abs_tol=1e-5)
    assert pitch == 0.0

    # Next tick advances by another 0.01
    roll, _ = filter_.process(target_roll=0.20, target_pitch=0.0, target_vx=0.0)
    assert math.isclose(roll, 0.02, abs_tol=1e-5)


def test_attitude_filter_hard_clamp():
    # max_roll_rad = 0.20
    filter_ = AttitudeCommandFilter(dt=0.02, max_rate_rad_s=100.0, max_forward_vel=1.0, max_roll_rad=0.10)

    # Command beyond 0.10 rad hard limit
    roll, _ = filter_.process(target_roll=0.50, target_pitch=0.0, target_vx=0.0)
    assert math.isclose(roll, 0.10, abs_tol=1e-5)


def test_microduck_locomotion_engine_contract():
    import numpy as np
    from microduck_brain.locomotion_engine import MicroduckLocomotionEngine, DEFAULT_POSE

    engine = MicroduckLocomotionEngine(onnx_model_path="models/microduck_walk.onnx")
    assert engine.num_dofs == 14
    assert engine.obs_dim == 61
    assert len(engine.default_dof_pos) == 14
    assert np.allclose(engine.default_dof_pos, DEFAULT_POSE)

    # Test 5D raw command input (vx, vy, wz, roll, pitch)
    raw_cmd = (0.15, 0.0, 0.0, 0.05, 0.0)
    grav = [0.0, 0.0, -1.0]
    ang_vel = [0.0, 0.0, 0.0]
    q_pos = DEFAULT_POSE.copy()
    q_vel = np.zeros(14, dtype=np.float32)

    pd_targets = engine.step(
        raw_cmd=raw_cmd,
        projected_gravity=grav,
        base_ang_vel=ang_vel,
        joint_pos=q_pos,
        joint_vel=q_vel,
    )

    assert len(pd_targets) == 14
    assert pd_targets.dtype == np.float32
    assert not np.any(np.isnan(pd_targets))
    assert len(engine.last_action) == 14
