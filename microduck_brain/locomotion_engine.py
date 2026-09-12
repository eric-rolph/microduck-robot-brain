"""
Locomotion controller and command conditioning filter for Tier 4.
Applies slew-rate limiting and velocity-dependent attenuation to body attitude trims,
assembles the 48D observation vector, and evaluates the ONNX locomotion policy.
"""

from __future__ import annotations
import math
import os
from typing import Optional, Sequence, Tuple
import numpy as np


class AttitudeCommandFilter:
    """
    Applies slew-rate limiting and velocity-dependent attenuation
    to incoming roll and pitch commands.
    """

    def __init__(
        self,
        dt: float = 0.02,                   # 50 Hz control period
        max_rate_rad_s: float = 0.5,        # Max ~28.6 deg/s
        max_forward_vel: float = 0.8,       # Velocity where attitude trims scale to 0
        max_roll_rad: float = 0.20,         # ~11.5 deg hard clamp
        max_pitch_rad: float = 0.20,
    ) -> None:
        self.dt = dt
        self.max_step = max_rate_rad_s * dt
        self.max_forward_vel = max_forward_vel
        self.max_roll = max_roll_rad
        self.max_pitch = max_pitch_rad

        # Filtered state: [roll, pitch]
        self.filtered_attitude = np.zeros(2, dtype=np.float32)

    def process(
        self,
        target_roll: float,
        target_pitch: float,
        target_vx: float,
    ) -> Tuple[float, float]:
        """
        1. Attenuates attitude commands based on forward speed.
        2. Clamps delta by max angular slew rate.
        """
        # Attenuate based on commanded forward speed
        speed_factor = max(0.0, 1.0 - (abs(target_vx) / self.max_forward_vel))
        clamped_roll = float(np.clip(target_roll, -self.max_roll, self.max_roll) * speed_factor)
        clamped_pitch = float(np.clip(target_pitch, -self.max_pitch, self.max_pitch) * speed_factor)
        targets = np.array([clamped_roll, clamped_pitch], dtype=np.float32)

        # Apply slew-rate clamping
        delta = targets - self.filtered_attitude
        slew_limited_delta = np.clip(delta, -self.max_step, self.max_step)
        self.filtered_attitude += slew_limited_delta

        return float(self.filtered_attitude[0]), float(self.filtered_attitude[1])

    def compute_safe_velocity(
        self,
        target_vx: float,
        measured_roll: float,
        measured_pitch: float,
        tilt_limit_rad: float = 0.26,  # ~15 degrees
    ) -> float:
        """
        Attenuates forward velocity when robot attitude tilts away from upright.
        Prevents pitching tumble falls by slowing down or stopping before balance is lost.
        """
        tilt = math.sqrt(measured_roll**2 + measured_pitch**2)
        if tilt >= tilt_limit_rad:
            return 0.0
        scale = max(0.0, 1.0 - (tilt / tilt_limit_rad))
        return float(target_vx * scale)

    def reset(self) -> None:
        self.filtered_attitude.fill(0.0)



# Default STAND2 pose for Microduck biped (14-DOF)
DEFAULT_POSE = np.array([
    0.0,      # left_hip_yaw
    -0.0873,  # left_hip_roll
    -0.4579,  # left_hip_pitch
    -0.0049,  # left_knee
    0.4530,   # left_ankle
    0.3491,   # neck_pitch
    0.3491,   # head_pitch
    0.0,      # head_yaw
    0.0,      # head_roll
    0.0,      # right_hip_yaw
    0.0873,   # right_hip_roll
    0.4579,   # right_hip_pitch
    0.0049,   # right_knee
    -0.4530,  # right_ankle
], dtype=np.float32)


class MicroduckLocomotionEngine:
    """
    Evaluates the 61-D ONNX biped policy at 50 Hz.
    Outputs 14-DOF PD position targets.
    """

    def __init__(
        self,
        onnx_model_path: Optional[str] = None,
        dt: float = 0.02,
    ) -> None:
        self.dt = dt
        self.attitude_filter = AttitudeCommandFilter(dt=dt, max_rate_rad_s=0.5)

        # Nominal standing joint angles (14-DOF biped)
        self.default_dof_pos = DEFAULT_POSE.copy()

        self.action_scale = 1.0
        self.num_dofs = 14
        self.obs_dim = 61
        self.last_action = np.zeros(self.num_dofs, dtype=np.float32)
        self.session = None
        self.input_name = None

        if onnx_model_path and os.path.exists(onnx_model_path):
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(onnx_model_path, opts, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name

            try:
                meta = self.session.get_modelmeta()
                if hasattr(meta, "custom_metadata_map") and "action_scale" in meta.custom_metadata_map:
                    self.action_scale = float(meta.custom_metadata_map["action_scale"])
            except Exception:
                pass

    def step(
        self,
        raw_cmd: Tuple[float, float, float, float, float] | Sequence[float],  # (vx, vy, wz, roll_cmd, pitch_cmd) or 13D
        projected_gravity: Sequence[float],                                 # [gx, gy, gz]
        base_ang_vel: Sequence[float],                                      # [wx, wy, wz]
        joint_pos: Sequence[float],                                         # 14D current angles
        joint_vel: Sequence[float],                                         # 14D current velocities
        base_lin_vel: Optional[Sequence[float]] = None,                     # [vx, vy, vz] optional
    ) -> np.ndarray:
        """
        Runs one 50 Hz control cycle and returns 14 PD setpoints matching the 61-D contract.
        """
        if len(raw_cmd) == 5:
            vx, vy, wz, raw_roll, raw_pitch = raw_cmd
            safe_roll, safe_pitch = self.attitude_filter.process(raw_roll, raw_pitch, vx)
            command = np.zeros(13, dtype=np.float32)
            command[0] = vx
            command[1] = vy
            command[2] = wz
            # Body pose roll/pitch in indices 7, 8
            command[7] = safe_roll
            command[8] = safe_pitch
        elif len(raw_cmd) == 13:
            command = np.asarray(raw_cmd, dtype=np.float32).copy()
        elif len(raw_cmd) == 3:
            vx, vy, wz = raw_cmd
            command = np.zeros(13, dtype=np.float32)
            command[0] = vx
            command[1] = vy
            command[2] = wz
        else:
            raise ValueError(f"Expected command length 3, 5, or 13, got {len(raw_cmd)}")

        # Build normalized relative proprioceptive vector
        joint_pos_arr = np.asarray(joint_pos, dtype=np.float32)
        joint_vel_arr = np.asarray(joint_vel, dtype=np.float32)
        joint_pos_rel = joint_pos_arr - self.default_dof_pos

        # Standard 61-D observation vector:
        # [base_ang_vel (3), projected_gravity (3), q_rel (14), joint_vel (14), last_action (14), command (13)]
        observation = np.concatenate([
            np.asarray(base_ang_vel, dtype=np.float32)[:3],
            np.asarray(projected_gravity, dtype=np.float32)[:3],
            joint_pos_rel[:14],
            joint_vel_arr[:14],
            self.last_action[:14],
            command[:13],
        ]).astype(np.float32).reshape(1, -1)

        # Model inference or fallback zero-action
        if self.session is not None and self.input_name is not None:
            outputs = self.session.run(None, {self.input_name: observation})
            action = outputs[0][0][:14]
        else:
            action = np.zeros(self.num_dofs, dtype=np.float32)

        self.last_action = action.copy()
        target_joint_angles = self.default_dof_pos + (action * self.action_scale)
        return target_joint_angles
