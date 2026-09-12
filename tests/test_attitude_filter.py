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
