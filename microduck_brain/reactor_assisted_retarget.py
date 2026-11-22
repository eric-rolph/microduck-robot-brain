"""Stage-1 Physics-Guided Motion Retargeting (ReActor, arXiv:2605.06593v1) on ExactBamPlant.

Implements the two key Stage-1 mechanisms from Mueller et al. (2026) adapted to the
14-DoF Microduck BAM M6 plant:
  1. **Virtual Root Residual Force Control (RFC)** with L1 sparsity deadband:
     Applies a bounded external stabilizing wrench ``w_k in R^6`` at the trunk only when
     needed to keep a kinematically retargeted stepping/kicking clip balanced on the plant,
     while progressively annealing the wrench schedule across adaptation passes.
  2. **Test-Time State Adaptation (TTSA)**:
     Iteratively updates per-frame joint reference offsets
     ``Delta_q[k] <- clip(q_exec[k] - q_kin[k], -delta_max, +delta_max)``
     and lead-compensated servo commands ``commands[k]`` so the resulting
     ``assisted_reference.npz`` is dynamically realized on the MuJoCo + BAM M6 plant and
     ready for Stage-2 force-free ``residual61`` policy tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .reactor_onnx_compiler import OFFICIAL_DEFAULT_POSE, OFFICIAL_JOINT_NAMES
from .sim.exact_bam_plant import CONTROL_DT_S, ExactBamPlant


@dataclass
class AssistedRetargetConfig:
    """Configuration for ReActor Stage-1 physics-guided assisted retargeting."""

    lead_controls: int = 2
    adaptation_passes: int = 3
    ttsa_max_offset_rad: float = 0.20
    ttsa_step_gain: float = 0.50
    pd_pos_gain_n_per_m: float = 120.0
    pd_vel_gain_n_s_per_m: float = 18.0
    pd_rot_gain_nm_per_rad: float = 4.5
    pd_angvel_gain_nm_s_per_rad: float = 0.45
    wrench_deadband_n: float = 0.15
    max_force_bw: float = 1.5
    max_torque_bwm: float = 0.25


def _quat_orientation_error_world(q_curr: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Compute 3D world-frame orientation error vector from q_curr to q_ref ([w, x, y, z])."""
    R_curr = np.empty(9, dtype=np.float64)
    R_ref = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(R_curr, np.asarray(q_curr, dtype=np.float64))
    mujoco.mju_quat2Mat(R_ref, np.asarray(q_ref, dtype=np.float64))
    R_c = R_curr.reshape(3, 3)
    R_r = R_ref.reshape(3, 3)
    # 0.5 * sum_i (R_curr[:, i] x R_ref[:, i])
    err = 0.5 * (
        np.cross(R_c[:, 0], R_r[:, 0])
        + np.cross(R_c[:, 1], R_r[:, 1])
        + np.cross(R_c[:, 2], R_r[:, 2])
    )
    return err


def _apply_deadband(vec: np.ndarray, deadband: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= deadband or norm < 1e-12:
        return np.zeros_like(vec)
    return vec * ((norm - deadband) / norm)


def run_assisted_retarget(
    kinematic_qpos: np.ndarray,
    *,
    source_frame_index: np.ndarray | None = None,
    stance_reference: np.ndarray | None = None,
    config: AssistedRetargetConfig | None = None,
    output_npz_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run Stage-1 ReActor assisted retargeting (RFC + TTSA) on a kinematic qpos trajectory.

    Parameters
    ----------
    kinematic_qpos : np.ndarray
        Array of shape ``(N, 21)`` containing ``[root_pos(3), root_quat(4), joints(14)]``.
    source_frame_index : np.ndarray | None
        Optional segment labels of length ``N``.
    stance_reference : np.ndarray | None
        Optional stance flags of shape ``(N, 2)``.
    config : AssistedRetargetConfig | None
        Hyperparameters for RFC wrench schedule and TTSA offset adaptation.
    output_npz_path : str | Path | None
        If provided, saves the resulting ``assisted_reference.npz`` archive.
    """
    cfg = config or AssistedRetargetConfig()
    q_kin = np.asarray(kinematic_qpos, dtype=np.float64)
    if q_kin.ndim != 2 or q_kin.shape[1] != 21 or len(q_kin) < 2:
        raise ValueError(f"Expected kinematic_qpos of shape (N>=2, 21), got {q_kin.shape}")

    N = len(q_kin)
    T = N - 1
    if source_frame_index is None:
        source_frame_index = np.arange(N, dtype=np.int64)
    if stance_reference is None:
        stance_reference = np.ones((N, 2), dtype=bool)

    plant = ExactBamPlant()
    weight_n = float(plant.model.body_subtreemass[0] * (-plant.model.opt.gravity[2]))

    # Finite-difference kinematic velocities
    qvel_kin = np.zeros((N, plant.model.nv), dtype=np.float64)
    for k in range(T):
        mujoco.mj_differentiatePos(plant.model, qvel_kin[k], CONTROL_DT_S, q_kin[k], q_kin[k + 1])
    qvel_kin[-1] = qvel_kin[-2]

    ttsa_offsets = np.zeros((N, 14), dtype=np.float64)
    best_rollout: dict[str, Any] | None = None

    for pass_idx in range(max(cfg.adaptation_passes, 1)):
        # Anneal wrench assistance across passes while TTSA offsets absorb feasible corrections
        assist_scale = max(0.25, 1.0 - 0.25 * pass_idx)
        adapted_joints = q_kin[:, 7:] + ttsa_offsets
        commands = np.stack(
            [adapted_joints[min(k + cfg.lead_controls, N - 1)] for k in range(T)],
            axis=0,
        )

        plant.reset_stand(qpos=q_kin[0])
        qpos_exec = [plant.data.qpos.copy()]
        qvel_exec = [plant.data.qvel.copy()]
        feet_exec = [plant.data.site_xpos[plant.foot_sites].copy()]
        wrenches = []
        fell_at = -1

        for k in range(T):
            pos_err = q_kin[k, 0:3] - plant.data.qpos[0:3]
            vel_err = qvel_kin[k, 0:3] - plant.data.qvel[0:3]
            rot_err = _quat_orientation_error_world(plant.data.qpos[3:7], q_kin[k, 3:7])
            ang_err = qvel_kin[k, 3:6] - plant.data.qvel[3:6]

            raw_force = assist_scale * (
                cfg.pd_pos_gain_n_per_m * pos_err + cfg.pd_vel_gain_n_s_per_m * vel_err
            )
            raw_torque = assist_scale * (
                cfg.pd_rot_gain_nm_per_rad * rot_err + cfg.pd_angvel_gain_nm_s_per_rad * ang_err
            )
            force = _apply_deadband(raw_force, cfg.wrench_deadband_n)
            torque = _apply_deadband(raw_torque, cfg.wrench_deadband_n * 0.05)

            f_max = cfg.max_force_bw * weight_n
            t_max = cfg.max_torque_bwm * weight_n
            f_norm = float(np.linalg.norm(force))
            if f_norm > f_max > 0:
                force = force * (f_max / f_norm)
            t_norm = float(np.linalg.norm(torque))
            if t_norm > t_max > 0:
                torque = torque * (t_max / t_norm)

            wrench_k = np.concatenate([force, torque]).astype(np.float64)
            wrenches.append(wrench_k)

            plant.data.xfrc_applied[plant.trunk, :3] = force
            plant.data.xfrc_applied[plant.trunk, 3:6] = torque
            plant.step_control(commands[k])
            plant.data.xfrc_applied[plant.trunk, :] = 0.0

            qpos_exec.append(plant.data.qpos.copy())
            qvel_exec.append(plant.data.qvel.copy())
            feet_exec.append(plant.data.site_xpos[plant.foot_sites].copy())

            if plant.is_fallen() and fell_at < 0:
                fell_at = k

        qpos_arr = np.asarray(qpos_exec, dtype=np.float64)
        qvel_arr = np.asarray(qvel_exec, dtype=np.float64)
        feet_arr = np.asarray(feet_exec, dtype=np.float64)
        wrench_arr = np.asarray(wrenches, dtype=np.float64)

        # Update TTSA offsets toward physically realized joint trajectory:
        # Delta_q[k] <- clip(Delta_q[k] + gain * (q_exec[k] - q_kin[k] - Delta_q[k]), -delta_max, +delta_max)
        realized_delta = qpos_arr[:, 7:] - q_kin[:, 7:]
        ttsa_offsets = np.clip(
            (1.0 - cfg.ttsa_step_gain) * ttsa_offsets + cfg.ttsa_step_gain * realized_delta,
            -cfg.ttsa_max_offset_rad,
            cfg.ttsa_max_offset_rad,
        )

        joint_rmse = float(np.sqrt(np.mean((qpos_arr[:, 7:] - q_kin[:, 7:]) ** 2)))
        mean_force_bw = float(np.linalg.norm(wrench_arr[:, :3], axis=1).mean() / max(weight_n, 1e-6))
        best_rollout = {
            "qpos": qpos_arr,
            "qvel": qvel_arr,
            "commands": commands,
            "foot_site_pos_w": feet_arr,
            "wrench": wrench_arr,
            "ttsa_offsets": ttsa_offsets.copy(),
            "kinematic_qpos": q_kin,
            "source_frame_index": np.asarray(source_frame_index, dtype=np.int64),
            "stance_reference": np.asarray(stance_reference, dtype=bool),
            "fps": np.array(50.0),
            "joint_names": np.array(OFFICIAL_JOINT_NAMES),
            "completed": bool(fell_at < 0),
            "fell_at_control": int(fell_at),
            "joint_rmse_rad": joint_rmse,
            "mean_force_bw": mean_force_bw,
            "default_pose": OFFICIAL_DEFAULT_POSE.copy(),
        }

    assert best_rollout is not None
    if output_npz_path is not None:
        out_p = Path(output_npz_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_p,
            qpos=best_rollout["qpos"],
            qvel=best_rollout["qvel"],
            commands=best_rollout["commands"],
            foot_site_pos_w=best_rollout["foot_site_pos_w"],
            wrench=best_rollout["wrench"],
            ttsa_offsets=best_rollout["ttsa_offsets"],
            kinematic_qpos=best_rollout["kinematic_qpos"],
            source_frame_index=best_rollout["source_frame_index"],
            stance_reference=best_rollout["stance_reference"],
            fps=best_rollout["fps"],
            joint_names=best_rollout["joint_names"],
        )
    return best_rollout
