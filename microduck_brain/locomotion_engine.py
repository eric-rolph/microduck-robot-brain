"""
Locomotion controller and command conditioning filter for Tier 4.
Applies slew-rate limiting and velocity-dependent attenuation to body attitude trims,
assembles the 48D observation vector, and evaluates the ONNX locomotion policy.
"""

from __future__ import annotations
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

    def reset(self) -> None:
        self.filtered_attitude.fill(0.0)


class MicroduckLocomotionEngine:
    """
    Evaluates the 48D ONNX quadruped policy at 50 Hz.
    Outputs 12-DOF PD position targets.
    """

    def __init__(
        self,
        onnx_model_path: Optional[str] = None,
        dt: float = 0.02,
    ) -> None:
        self.dt = dt
        self.attitude_filter = AttitudeCommandFilter(dt=dt, max_rate_rad_s=0.5)

        # Nominal standing joint angles (12-DOF)
        # FL: Hip, Thigh, Calf; FR; RL; RR
        self.default_dof_pos = np.array([
             0.1,  0.8, -1.5,   # FL
            -0.1,  0.8, -1.5,   # FR
             0.1,  1.0, -1.5,   # RL
            -0.1,  1.0, -1.5,   # RR
        ], dtype=np.float32)

        self.action_scale = 0.25
        self.last_action = np.zeros(12, dtype=np.float32)
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

    def step(
        self,
        raw_cmd: Tuple[float, float, float, float, float],  # (vx, vy, wz, roll_cmd, pitch_cmd)
        projected_gravity: Sequence[float],                # [gx, gy, gz]
        base_lin_vel: Sequence[float],                     # [vx, vy, vz]
        base_ang_vel: Sequence[float],                     # [wx, wy, wz]
        joint_pos: Sequence[float],                        # 12D current angles
        joint_vel: Sequence[float],                        # 12D current velocities
    ) -> np.ndarray:
        """
        Runs one 50 Hz control cycle and returns 12 PD setpoints.
        """
        vx, vy, wz, raw_roll, raw_pitch = raw_cmd

        # 1. Rate-limit and attenuate attitude commands
        safe_roll, safe_pitch = self.attitude_filter.process(raw_roll, raw_pitch, vx)
        filtered_commands = np.array([vx, vy, wz, safe_roll, safe_pitch], dtype=np.float32)

        # 2. Build normalized relative proprioceptive vector
        joint_pos_arr = np.asarray(joint_pos, dtype=np.float32)
        joint_vel_arr = np.asarray(joint_vel, dtype=np.float32)
        joint_pos_rel = joint_pos_arr - self.default_dof_pos

        # Observation shape: 48D
        observation = np.concatenate([
            np.asarray(projected_gravity, dtype=np.float32),
            np.asarray(base_lin_vel, dtype=np.float32),
            np.asarray(base_ang_vel, dtype=np.float32),
            joint_pos_rel,
            joint_vel_arr * 0.05,
            self.last_action,
            filtered_commands,
        ]).astype(np.float32).reshape(1, -1)

        # 3. Model inference or fallback heuristic
        if self.session is not None and self.input_name is not None:
            outputs = self.session.run(None, {self.input_name: observation})
            action = outputs[0][0]
        else:
            # Fallback zero-offset policy for offline testing
            action = np.zeros(12, dtype=np.float32)

        self.last_action = action.copy()
        target_joint_angles = self.default_dof_pos + (action * self.action_scale)
        return target_joint_angles
