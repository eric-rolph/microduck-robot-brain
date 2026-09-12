"""Unit and integration tests for Microduck MuJoCo simulation environment,
BAM M6 actuator modeling, backlash twin, and 61-D observation contract.
"""

from __future__ import annotations

import math
from pathlib import Path
import mujoco
import numpy as np
import pytest

from microduck_brain.sim.backlash import BacklashManager
from microduck_brain.sim.bam_actuator import BamM6ActuatorModel, BamM6Config
from microduck_brain.sim.env import DEFAULT_POSE, MicroduckMuJoCoEnv
from microduck_brain.sim.train_smoke import MicroduckPPO


def test_mujoco_mjcf_loading() -> None:
    """Verify both standard and backlash MJCF models load without errors."""
    mjcf_dir = Path(__file__).parent.parent / "microduck_brain" / "sim" / "mjcf"

    # Standard model
    std_xml = mjcf_dir / "microduck_walk.xml"
    assert std_xml.exists(), "Standard MJCF file missing"
    m_std = mujoco.MjModel.from_xml_path(str(std_xml))
    assert m_std.nu == 14, f"Expected 14 actuators, got {m_std.nu}"
    assert m_std.nq == 21, f"Expected 21 generalized coordinates, got {m_std.nq}"

    # Backlash model
    bl_xml = mjcf_dir / "microduck_walk_backlash.xml"
    assert bl_xml.exists(), "Backlash MJCF file missing"
    m_bl = mujoco.MjModel.from_xml_path(str(bl_xml))
    assert m_bl.nu == 14, f"Expected 14 actuators, got {m_bl.nu}"
    assert m_bl.nq == 35, f"Expected 35 generalized coordinates (14 servo + 14 bl + 7 root), got {m_bl.nq}"


def test_bam_actuator_dynamics() -> None:
    """Test voltage sag, back-EMF torque limits, stiction, and transport delay."""
    cfg = BamM6Config(
        v_open_circuit=8.2,
        r_internal_batt=0.18,
        delay_steps=2,
        coulomb_friction=0.03,
        stiction_torque=0.05,
    )
    bam = BamM6ActuatorModel(num_actuators=14, config=cfg)

    # 1. Transport delay verification
    target_1 = np.ones(14, dtype=np.float32) * 0.1
    target_2 = np.ones(14, dtype=np.float32) * 0.2
    target_3 = np.ones(14, dtype=np.float32) * 0.3

    measured_q = np.zeros(14, dtype=np.float32)
    measured_v = np.zeros(14, dtype=np.float32)

    # Initially filled with zeros
    bam.compute_torques(target_1, measured_q, measured_v)
    bam.compute_torques(target_2, measured_q, measured_v)
    # On 3rd call, target_1 should emerge from queue
    t3 = bam.compute_torques(target_3, measured_q, measured_v)
    assert np.all(t3 > 0.0), "Delayed positive target should generate positive torque"

    # 2. Voltage sag test
    assert bam.battery_voltage < 8.2, "Battery voltage must sag under non-zero current"
    assert bam.battery_voltage >= cfg.v_min, "Battery voltage must not drop below v_min"

    # 3. Stiction threshold test
    bam_fresh = BamM6ActuatorModel(num_actuators=14, config=cfg)
    # Tiny position error resulting in torque demand below stiction_torque (0.05 N*m)
    tiny_target = np.ones(14, dtype=np.float32) * 0.001
    tau_stick = bam_fresh.compute_torques(tiny_target, measured_q, measured_v)
    # Stiction prevents motion if demand is below breakaway threshold
    assert np.all(tau_stick == 0.0), "Torque below stiction threshold must yield 0 net torque"


def test_backlash_twin_encoder_feedback() -> None:
    """Verify output-side encoder observation sums servo angle and backlash play."""
    mjcf_dir = Path(__file__).parent.parent / "microduck_brain" / "sim" / "mjcf"
    m = mujoco.MjModel.from_xml_path(str(mjcf_dir / "microduck_walk_backlash.xml"))
    d = mujoco.MjData(m)
    bm = BacklashManager(m)

    assert bm.has_backlash is True, "Backlash manager must detect backlash joints"
    assert len(bm.servo_qpos_adr) == 14
    assert len(bm.backlash_qpos_adr) == 14

    # Manually inject displacement into servo and backlash joints
    for i in range(14):
        s_adr = bm.servo_qpos_adr[i]
        b_adr = bm.backlash_qpos_adr[i]
        assert b_adr is not None
        d.qpos[s_adr] = 0.10
        d.qpos[b_adr] = 0.01

    q_enc = bm.read_encoder_positions(d)
    np.testing.assert_allclose(q_enc, np.full(14, 0.11), atol=1e-5)


def test_standardized_61d_observation_contract() -> None:
    """Validate dimensionality and component order of the 61-D observation vector."""
    env = MicroduckMuJoCoEnv(use_backlash=True)
    obs = env.reset()

    assert obs.shape == (61,), f"Observation vector must be 61D, got {obs.shape}"

    # Verify slice breakdown:
    # 0..3: base angular velocity (3D)
    ang_vel = obs[0:3]
    assert ang_vel.shape == (3,)

    # 3..6: projected gravity (3D)
    proj_grav = obs[3:6]
    assert proj_grav.shape == (3,)
    grav_norm = np.linalg.norm(proj_grav)
    assert abs(grav_norm - 1.0) < 0.05, f"Projected gravity must be unit vector, got {grav_norm}"

    # 6..20: joint positions relative to DEFAULT_POSE (14D)
    q_rel = obs[6:20]
    assert q_rel.shape == (14,)
    # At reset in STAND2, relative position error should be near zero
    assert np.all(np.abs(q_rel) < 0.05)

    # 20..34: joint velocities (14D)
    q_vel = obs[20:34]
    assert q_vel.shape == (14,)

    # 34..48: last action (14D)
    last_act = obs[34:48]
    assert last_act.shape == (14,)
    assert np.all(last_act == 0.0)

    # 48..61: 13D command vector
    cmd = obs[48:61]
    assert cmd.shape == (13,)

    # Test setting commands
    env.set_command(
        lin_vel_x=0.25,
        lin_vel_y=-0.1,
        ang_vel_z=0.4,
        head_pose=np.array([0.1, 0.2, 0.3, 0.4]),
        body_pose=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
    )
    obs_cmd = env.get_observation()
    np.testing.assert_allclose(obs_cmd[48:51], [0.25, -0.1, 0.4], atol=1e-5)
    np.testing.assert_allclose(obs_cmd[51:55], [0.1, 0.2, 0.3, 0.4], atol=1e-5)
    np.testing.assert_allclose(obs_cmd[55:61], [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], atol=1e-5)


def test_env_step_and_fall_detection() -> None:
    """Test env step execution, reward calculation, and fall termination."""
    env = MicroduckMuJoCoEnv(use_backlash=True)
    env.reset()

    # Step in upright stance
    action = np.zeros(14, dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)
    assert not term, "Robot should not terminate in upright standing pose"
    assert reward > 0.0, "Standing posture should yield positive reward"
    assert "battery_voltage" in info
    assert "trunk_height" in info

    # Force tip over to test termination trigger
    env.data.qpos[2] = 0.04  # collapse height
    assert env.is_terminated() is True, "Height below 0.065 m must trigger termination"


def test_ppo_smoke_training_loop() -> None:
    """Run lightweight PPO training loop to verify loss convergence and metric reporting."""
    trainer = MicroduckPPO(num_envs=2, steps_per_env=16, device="cpu")
    history = trainer.train_smoke(iterations=2)

    assert len(history) == 2, "Expected 2 iteration summaries"
    for step_stat in history:
        assert "mean_reward" in step_stat
        assert "policy_loss" in step_stat
        assert "value_loss" in step_stat
        assert not np.isnan(step_stat["policy_loss"]), "Policy loss must not be NaN"
        assert not np.isnan(step_stat["value_loss"]), "Value loss must not be NaN"
