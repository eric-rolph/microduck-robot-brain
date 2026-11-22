"""
Locomotion controller, command conditioning filter, and closed-loop 61-D ReActor
skill/recovery supervisor for Tier 4.
Applies slew-rate limiting and velocity-dependent attenuation to body attitude trims,
assembles the 61-D observation vector, evaluates the ONNX locomotion policy,
and orchestrates seamless handovers between walking, closed-loop 61-D ReActor skills
(bow, kick, wave, call, karate_kicks, jazzwalk), and autonomous get-up recovery.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .mocap_engine import ReactorSkillPlayer
from .reactor_onnx_compiler import OFFICIAL_DEFAULT_POSE


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


# Default STAND2 pose for Microduck biped (14-DOF), exact 7-decimal official contract
DEFAULT_POSE = OFFICIAL_DEFAULT_POSE.copy()


class MicroduckLocomotionEngine:
    """
    Evaluates the 61-D ONNX biped policy at 50 Hz.
    Outputs 14-DOF PD position targets and coordinates closed-loop ReActor skills
    and autonomous get-up recovery.
    """

    def __init__(
        self,
        onnx_model_path: Optional[str] = None,
        dt: float = 0.02,
        stand_recovery_onnx_path: Optional[str] = None,
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

        # Registered ReActor 61-D ONNX skills and stand recovery policy
        self.skills: Dict[str, ReactorSkillPlayer] = {}
        self.active_skill_name: Optional[str] = None
        self.stand_session = None
        self.stand_input_name: Optional[str] = None
        self.stand_action_scale: float = 1.0

        # Recovery supervisor state
        self.recovery_mode: str = "normal"  # "normal" | "settling" | "get_up"
        self.fallen_counter: int = 0
        self.settle_counter: int = 0
        self.upright_counter: int = 0

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

        if stand_recovery_onnx_path and os.path.exists(stand_recovery_onnx_path):
            self.register_stand_recovery(stand_recovery_onnx_path)

    def register_stand_recovery(self, onnx_path: str | Path) -> None:
        """Load the 61-D get-up recovery policy (e.g. BEST_alpha_stand.onnx)."""
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.stand_session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.stand_input_name = self.stand_session.get_inputs()[0].name
        self.stand_action_scale = 1.0
        try:
            meta = self.stand_session.get_modelmeta()
            if hasattr(meta, "custom_metadata_map") and "action_scale" in meta.custom_metadata_map:
                self.stand_action_scale = float(meta.custom_metadata_map["action_scale"])
        except Exception:
            pass

    def register_skill(
        self,
        skill_name: str,
        onnx_path: str | Path,
        *,
        blend_entry_frames: int = 15,
    ) -> ReactorSkillPlayer:
        """Register a self-contained 61-D -> 14-D ReActor ONNX skill."""
        player = ReactorSkillPlayer(onnx_path, blend_entry_frames=blend_entry_frames)
        self.skills[skill_name] = player
        return player

    def register_default_skills(
        self,
        models_dir: str | Path = "models",
        *,
        include_experimental: bool = False,
    ) -> list[str]:
        """Auto-discover and register verified 61-D ONNX skills in models_dir.

        Reads ``manifest.json`` when present and skips skills tagged
        ``experimental_failing_fidelity`` unless ``include_experimental=True``.
        """
        import json

        mdir = Path(models_dir)
        manifest_status: dict[str, str] = {}
        manifest_path = mdir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for entry in manifest.get("policies", []):
                    key = entry.get("name", Path(entry["file"]).stem)
                    if "status" in entry:
                        manifest_status[key] = str(entry["status"])
            except Exception:
                pass

        registered: list[str] = []
        for name in ("bow", "bow_walk_entry", "kick", "wave", "byebye", "call", "karate_kicks", "jazzwalk"):
            status = manifest_status.get(name, "")
            if status == "experimental_failing_fidelity" and not include_experimental:
                continue
            p = mdir / f"{name}.onnx"
            if p.is_file():
                self.register_skill(name, p)
                registered.append(name)
        stand_path = mdir / "BEST_alpha_stand.onnx"
        if stand_path.is_file() and self.stand_session is None:
            self.register_stand_recovery(stand_path)
        return registered

    def trigger_skill(
        self,
        skill_name: str,
        *,
        initial_joint_vel: Optional[Sequence[float]] = None,
    ) -> bool:
        """Start execution of a registered ReActor ONNX skill from the current action state."""
        if skill_name not in self.skills:
            return False
        player = self.skills[skill_name]
        player.start(
            entry_action=self.last_action.copy(),
            initial_joint_vel=np.asarray(initial_joint_vel, dtype=np.float32)
            if initial_joint_vel is not None
            else None,
        )
        self.active_skill_name = skill_name
        return True

    def declare_fall(self) -> None:
        """Enter settling -> get-up recovery mode."""
        self.active_skill_name = None
        self.recovery_mode = "settling"
        self.settle_counter = 0
        self.upright_counter = 0

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
        If a ReActor skill or stand recovery is active, executes that policy and hands
        last_action seamlessly back to the walker upon completion.
        """
        _ = base_lin_vel
        grav_arr = np.asarray(projected_gravity, dtype=np.float32)[:3]
        gyro_arr = np.asarray(base_ang_vel, dtype=np.float32)[:3]
        joint_pos_arr = np.asarray(joint_pos, dtype=np.float32)[:14]
        joint_vel_arr = np.asarray(joint_vel, dtype=np.float32)[:14]

        # 1. Autonomous fall recovery supervisor (when stand_session is registered)
        if self.stand_session is not None:
            if self.recovery_mode == "normal":
                if float(grav_arr[2]) > -0.50:
                    self.fallen_counter += 1
                    if self.fallen_counter >= 5:
                        self.declare_fall()
                else:
                    self.fallen_counter = 0

            if self.recovery_mode == "settling":
                self.settle_counter += 1
                if self.settle_counter >= 15:
                    self.recovery_mode = "get_up"
                    self.last_action = np.zeros(self.num_dofs, dtype=np.float32)
                return self.default_dof_pos + (self.last_action * self.stand_action_scale)

            if self.recovery_mode == "get_up":
                obs_stand = np.concatenate([
                    gyro_arr,
                    grav_arr,
                    joint_pos_arr - self.default_dof_pos,
                    joint_vel_arr,
                    self.last_action[:14],
                    np.zeros(13, dtype=np.float32),
                ]).astype(np.float32).reshape(1, -1)
                act = self.stand_session.run(None, {self.stand_input_name: obs_stand})[0][0][:14]
                self.last_action = np.asarray(act, dtype=np.float32).copy()
                if float(grav_arr[2]) < -0.85:
                    self.upright_counter += 1
                    if self.upright_counter >= 50:
                        self.recovery_mode = "normal"
                        self.fallen_counter = 0
                else:
                    self.upright_counter = 0
                return self.default_dof_pos + (self.last_action * self.stand_action_scale)

        # 2. Closed-loop ReActor skill execution (when active)
        if self.active_skill_name is not None and self.active_skill_name in self.skills:
            player = self.skills[self.active_skill_name]
            finished, target_q, applied_act = player.step_closed_loop(
                gyro=gyro_arr,
                projected_gravity=grav_arr,
                joint_pos=joint_pos_arr,
                joint_vel=joint_vel_arr,
            )
            self.last_action = applied_act.copy()
            if finished:
                self.last_action = player.handover_last_action().copy()
                self.active_skill_name = None
            return target_q.astype(np.float32)

        # 3. Standard locomotion / posture command processing
        if len(raw_cmd) == 5:
            vx, vy, wz, raw_roll, raw_pitch = raw_cmd
            measured_roll = math.atan2(float(grav_arr[1]), max(1e-6, -float(grav_arr[2])))
            measured_pitch = math.atan2(-float(grav_arr[0]), max(1e-6, -float(grav_arr[2])))
            safe_vx = self.attitude_filter.compute_safe_velocity(vx, measured_roll, measured_pitch)
            safe_roll, safe_pitch = self.attitude_filter.process(raw_roll, raw_pitch, safe_vx)
            command = np.zeros(13, dtype=np.float32)
            command[0] = safe_vx
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
        joint_pos_rel = joint_pos_arr - self.default_dof_pos

        # Standard 61-D observation vector:
        # [base_ang_vel (3), projected_gravity (3), q_rel (14), joint_vel (14), last_action (14), command (13)]
        observation = np.concatenate([
            gyro_arr,
            grav_arr,
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
