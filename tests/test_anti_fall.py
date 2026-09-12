"""
Unit and integration tests for the anti-fall training and balance recovery system.
Validates terminal fall penalties, survival bonuses, non-foot collision gating,
push disturbance responses, proactive speed attenuation, and fall rate logging.
"""

import math
import numpy as np
import pytest

from microduck_brain.locomotion_engine import AttitudeCommandFilter
from microduck_brain.sim.env import MicroduckMuJoCoEnv
from microduck_brain.sim.train_smoke import MicroduckPPO
from microduck_brain.world_state import WorldState


def test_terminal_fall_penalty_applied():
    """Verify that falling incurs a -50.0 terminal penalty and upright stance gives positive reward."""
    env = MicroduckMuJoCoEnv(use_backlash=True)
    env.reset()

    # Upright standing step
    obs, reward_upright, term, trunc, info = env.step(np.zeros(14, dtype=np.float32))
    assert not term
    assert reward_upright > 1.0, f"Upright stance should yield positive reward, got {reward_upright}"

    # Force tip over (> 41 degrees)
    env.data.qpos[2] = 0.04  # collapsed height
    obs, reward_fall, term, trunc, info = env.step(np.zeros(14, dtype=np.float32))
    assert term is True, "Collapsed height must terminate episode"
    assert reward_fall == -50.0, f"Falling must incur -50.0 terminal penalty, got {reward_fall}"
    assert info["is_fall"] is True


def test_non_foot_ground_collision_detection():
    """Verify that non-foot collisions (trunk/knees on floor) trigger termination."""
    env = MicroduckMuJoCoEnv(use_backlash=True)
    env.reset()

    # Normal stance: no non-foot collision
    assert env.check_non_foot_ground_collision() is False

    # Move trunk directly onto the floor
    env.data.qpos[2] = 0.02
    import mujoco
    mujoco.mj_step(env.model, env.data)

    assert env.check_non_foot_ground_collision() is True
    assert env.is_terminated() is True


def test_push_disturbance_application():
    """Verify external velocity impulses perturb trunk velocity for recovery training."""
    env = MicroduckMuJoCoEnv(use_backlash=True)
    env.reset()

    vx_init = float(env.data.qvel[0])
    vy_init = float(env.data.qvel[1])

    delta_vx = 0.25
    delta_vy = -0.18
    env.apply_push_disturbance(delta_vx, delta_vy)

    assert abs(env.data.qvel[0] - (vx_init + delta_vx)) < 1e-5
    assert abs(env.data.qvel[1] - (vy_init + delta_vy)) < 1e-5


def test_proactive_safe_velocity_attenuation():
    """Verify forward speed is automatically scaled down under body tilt."""
    filter_node = AttitudeCommandFilter()

    # Upright: 0.5 m/s command remains 0.5 m/s
    v_safe = filter_node.compute_safe_velocity(target_vx=0.5, measured_roll=0.0, measured_pitch=0.0)
    assert abs(v_safe - 0.5) < 1e-5

    # Moderate tilt (~7.5 deg = 0.13 rad): attenuated to ~0.25 m/s
    v_attenuated = filter_node.compute_safe_velocity(target_vx=0.5, measured_roll=0.13, measured_pitch=0.0)
    assert 0.15 < v_attenuated < 0.35

    # Dangerous tilt (> 15 deg = 0.26 rad): speed clamped to 0.0 m/s
    v_halt = filter_node.compute_safe_velocity(target_vx=0.5, measured_roll=0.28, measured_pitch=0.0)
    assert v_halt == 0.0


def test_world_state_tilt_margin_and_near_fall():
    """Verify WorldState tracks tilt degrees and asserts is_near_fall above 25 deg."""
    ws = WorldState()

    # Upright
    ws.update_orientation(roll=0.0, pitch=0.0)
    assert ws.tilt_deg == 0.0
    assert ws.is_near_fall is False

    # Mild 10 degree tilt
    ws.update_orientation(roll=math.radians(10.0), pitch=0.0)
    assert abs(ws.tilt_deg - 10.0) < 0.1
    assert ws.is_near_fall is False

    # Near fall (28 degree tilt)
    ws.update_orientation(roll=math.radians(28.0), pitch=0.0)
    assert abs(ws.tilt_deg - 28.0) < 0.1
    assert ws.is_near_fall is True
    assert ws.to_dict()["is_near_fall"] is True


def test_anti_fall_training_fall_rate_metric():
    """Verify PPO trainer reports fall_rate and total_falls across iterations."""
    trainer = MicroduckPPO(num_envs=2, steps_per_env=16, device="cpu", enable_push_curriculum=True)
    history = trainer.train_smoke(iterations=2)

    assert len(history) == 2
    for metrics in history:
        assert "fall_rate" in metrics
        assert "total_falls" in metrics
        assert 0.0 <= metrics["fall_rate"] <= 1.0
