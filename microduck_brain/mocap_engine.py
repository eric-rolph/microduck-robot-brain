"""
Mocap playback, retargeting, and closed-loop 61-D ReActor ONNX skill engine
for the Microduck 14-DOF biped.

Supports both:
  1. Closed-loop physics-aware 61-D -> 14-D ReActor ONNX skills (``ReactorSkillPlayer``),
     executing self-contained ``official61`` policies with Fourier/finite phase encoding,
     1-control lagged joint velocity, smooth entry/exit transitions, and walker handover.
  2. Open-loop ``.npz`` clip playback (``MocapClip`` / ``MocapPlayer``) with stability-first
     projection around ``STAND_POSE`` as a fallback when no ONNX policy is used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .reactor_onnx_compiler import (
    OFFICIAL_DEFAULT_POSE,
    PHASE_SCHEMA_FOURIER_V1,
    compute_phase_command,
    read_reactor_onnx_metadata,
    transition_envelope,
)

# Standard Microduck 14-joint order:
# 0: left_hip_yaw, 1: left_hip_roll, 2: left_hip_pitch, 3: left_knee, 4: left_ankle,
# 5: neck_pitch, 6: head_pitch, 7: head_yaw, 8: head_roll,
# 9: right_hip_yaw, 10: right_hip_roll, 11: right_hip_pitch, 12: right_knee, 13: right_ankle
STAND_POSE = np.array([
    0.0, -0.087266, -0.457924, -0.00494, 0.452984,
    0.349066, 0.349066, 0.0, 0.0,
    0.0, 0.087266, 0.457924, 0.00494, -0.452984
], dtype=np.float64)

# Intent scaling factors: preserve head/neck expressive gesture while attenuating
# joint excursions to maintain center-of-mass inside the 1.35 cm sole contact patch
BOW_INTENT_SCALE = np.array([
    0.30, 0.30, 0.15, 0.00, 0.08,
    0.80, 0.70, 0.30, 0.20,
    0.30, 0.30, 0.15, 0.00, 0.08
], dtype=np.float64)


class MocapClip:
    """Represents a retargeted motion capture clip."""

    def __init__(self, clip_path: str | Path) -> None:
        path = Path(clip_path)
        if not path.is_file():
            raise FileNotFoundError(f"Mocap clip not found: {clip_path}")

        data = np.load(path)
        self.joint_pos: np.ndarray = data["joint_pos"]  # (N, 14)
        self.joint_names: list[str] = [str(name) for name in data["joint_names"]]
        self.fps: float = float(data["fps"][0]) if data["fps"].ndim > 0 else float(data["fps"])
        self.times_s: np.ndarray = data["times_s"]
        self.num_frames: int = len(self.joint_pos)
        self.duration_s: float = float(self.times_s[-1]) if self.num_frames > 0 else 0.0


class ReactorSkillPlayer:
    """Closed-loop 50 Hz player for self-contained 61-D -> 14-D ReActor ONNX skills.

    Evaluates a compiled ``official61`` ONNX policy on live 61-D proprioception +
    13-D finite-phase Fourier commands, maintaining the 1-control velocity lag and
    ``last_raw`` action history required by the trained tracking policy.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        blend_entry_frames: int = 15,
        entry_ramp_frames: int = 0,
        exit_ramp_frames: int = 0,
        frames_override: int | None = None,
        phase_schema_override: str | None = None,
    ) -> None:
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ReActor skill ONNX not found: {onnx_path}")

        self.metadata = read_reactor_onnx_metadata(self.onnx_path)
        self.skill_name = str(self.metadata.get("skill_name", self.onnx_path.stem))
        self.frames = int(frames_override or self.metadata.get("frames", 229))
        self.control_steps = int(self.metadata.get("control_steps", max(self.frames - 1, 1)))
        self.phase_schema = str(
            phase_schema_override or self.metadata.get("phase_schema", PHASE_SCHEMA_FOURIER_V1)
        )
        self.fps = float(self.metadata.get("fps", 50.0))
        self.default_pose = OFFICIAL_DEFAULT_POSE.astype(np.float64).copy()
        self.blend_entry_frames = int(blend_entry_frames)
        self.entry_ramp_frames = int(entry_ramp_frames)
        self.exit_ramp_frames = int(exit_ramp_frames)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        in_shape = self.session.get_inputs()[0].shape
        out_shape = self.session.get_outputs()[0].shape
        if in_shape[-1] != 61 or out_shape[-1] != 14:
            raise ValueError(
                f"Expected 61->14 self-contained ONNX policy, got {in_shape} -> {out_shape}"
            )

        self.current_frame = 0
        self.is_active = False
        self.prev_vel = np.zeros(14, dtype=np.float32)
        self.last_raw = np.zeros(14, dtype=np.float32)
        self.last_applied_action = np.zeros(14, dtype=np.float32)
        self.entry_action = np.zeros(14, dtype=np.float32)

    @property
    def num_frames(self) -> int:
        return self.control_steps

    def start(
        self,
        current_robot_pose: Optional[np.ndarray] = None,
        *,
        entry_action: Optional[np.ndarray] = None,
        initial_joint_vel: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize closed-loop skill episode."""
        self.current_frame = 0
        self.is_active = True
        self.last_raw = np.zeros(14, dtype=np.float32)
        if initial_joint_vel is not None:
            self.prev_vel = np.asarray(initial_joint_vel, dtype=np.float32).reshape(14).copy()
        else:
            self.prev_vel = np.zeros(14, dtype=np.float32)

        if entry_action is not None:
            self.entry_action = np.asarray(entry_action, dtype=np.float32).reshape(14).copy()
        elif current_robot_pose is not None and len(current_robot_pose) == 14:
            self.entry_action = (
                np.asarray(current_robot_pose, dtype=np.float64).reshape(14) - self.default_pose
            ).astype(np.float32)
        else:
            self.entry_action = np.zeros(14, dtype=np.float32)
        self.last_applied_action = self.entry_action.copy()

    def build_observation(
        self,
        gyro: Sequence[float],
        projected_gravity: Sequence[float],
        joint_pos: Sequence[float],
        joint_vel: Sequence[float],
        *,
        active_flag: float = 1.0,
    ) -> np.ndarray:
        """Assemble the 61-D observation vector for the current frame."""
        k = min(max(self.current_frame, 0), self.frames - 1)
        phase_13 = compute_phase_command(
            k, self.frames, phase_schema=self.phase_schema, active=active_flag
        )
        q_rel = np.asarray(joint_pos, dtype=np.float32).reshape(14) - OFFICIAL_DEFAULT_POSE
        obs = np.concatenate([
            np.asarray(gyro, dtype=np.float32).reshape(3),
            np.asarray(projected_gravity, dtype=np.float32).reshape(3),
            q_rel,
            self.prev_vel,
            self.last_raw,
            phase_13,
        ]).astype(np.float32)
        self.prev_vel = np.asarray(joint_vel, dtype=np.float32).reshape(14).copy()
        return obs

    def step_closed_loop(
        self,
        gyro: Sequence[float],
        projected_gravity: Sequence[float],
        joint_pos: Sequence[float],
        joint_vel: Sequence[float],
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Advance one 50 Hz closed-loop control step.

        Returns:
            (finished: bool, target_joint_angles_14: np.ndarray, applied_action_14: np.ndarray)
        """
        if not self.is_active or self.current_frame >= self.control_steps:
            self.is_active = False
            target = self.default_pose + self.last_applied_action.astype(np.float64)
            return True, target, self.last_applied_action.copy()

        psi = transition_envelope(
            self.current_frame,
            self.control_steps,
            entry_ramp_frames=self.entry_ramp_frames,
            exit_ramp_frames=self.exit_ramp_frames,
        )
        obs = self.build_observation(
            gyro, projected_gravity, joint_pos, joint_vel, active_flag=psi
        )
        raw_act = np.asarray(
            self.session.run(None, {self.input_name: obs[None, :]})[0][0][:14],
            dtype=np.float32,
        )
        self.last_raw = raw_act.copy()

        if self.blend_entry_frames > 0 and self.current_frame < self.blend_entry_frames:
            alpha = float(self.current_frame + 1) / float(self.blend_entry_frames)
            applied_act = (1.0 - alpha) * self.entry_action + alpha * raw_act
        else:
            applied_act = raw_act.copy()

        self.last_applied_action = applied_act.astype(np.float32)
        target_q = self.default_pose + applied_act.astype(np.float64)
        self.current_frame += 1
        finished = self.current_frame >= self.control_steps
        if finished:
            self.is_active = False
        return finished, target_q, self.last_applied_action.copy()

    def step(
        self, sensor_state: Optional[Mapping[str, Any]] = None
    ) -> Tuple[bool, np.ndarray]:
        """Compatibility step() method for Behavior Tree nodes."""
        if sensor_state is not None and all(
            k in sensor_state for k in ("gyro", "projected_gravity", "joint_pos", "joint_vel")
        ):
            done, target_q, _ = self.step_closed_loop(
                sensor_state["gyro"],
                sensor_state["projected_gravity"],
                sensor_state["joint_pos"],
                sensor_state["joint_vel"],
            )
            return done, target_q

        # Nominal upright sensor fallback when called without live sensors
        done, target_q, _ = self.step_closed_loop(
            gyro=np.zeros(3, dtype=np.float32),
            projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            joint_pos=(self.default_pose + self.last_applied_action.astype(np.float64)).astype(np.float32),
            joint_vel=np.zeros(14, dtype=np.float32),
        )
        return done, target_q

    def handover_last_action(self) -> np.ndarray:
        """Return the last default-pose-relative action for seamless walker handover."""
        return self.last_raw.copy()


class MocapPlayer:
    """
    50 Hz Mocap trajectory player with stability-first projection.
    Streams joint setpoints frame-by-frame, ensuring smooth blend from/to standing.
    Also provides ``MocapPlayer.from_onnx(...)`` to construct a closed-loop
    :class:`ReactorSkillPlayer` from a self-contained 61-D ONNX skill.
    """

    def __init__(
        self,
        clip: MocapClip,
        intent_scale: Optional[np.ndarray] = None,
        blend_frames: int = 15,
    ) -> None:
        self.clip = clip
        self.intent_scale = intent_scale if intent_scale is not None else BOW_INTENT_SCALE
        self.blend_frames = blend_frames
        self.current_frame = 0
        self.is_active = False
        self.start_pose = STAND_POSE.copy()

        # Precompute projected trajectory
        self.projected_trajectory = self._project_trajectory()

    @staticmethod
    def from_onnx(
        onnx_path: str | Path,
        *,
        blend_entry_frames: int = 15,
    ) -> ReactorSkillPlayer:
        """Load a closed-loop 61-D -> 14-D ReActor ONNX skill player."""
        return ReactorSkillPlayer(onnx_path, blend_entry_frames=blend_entry_frames)

    def _project_trajectory(self) -> np.ndarray:
        """Projects raw retargeted frames around STAND_POSE using intent scaling."""
        projected = np.zeros_like(self.clip.joint_pos)
        for i in range(self.clip.num_frames):
            raw = self.clip.joint_pos[i]
            # Offset from the clip's initial frame scaled by intent
            delta = (raw - self.clip.joint_pos[0]) * self.intent_scale
            projected[i] = STAND_POSE + delta
        return projected

    def start(self, current_robot_pose: Optional[np.ndarray] = None) -> None:
        """Starts playback blending from current robot pose."""
        self.current_frame = 0
        self.is_active = True
        if current_robot_pose is not None and len(current_robot_pose) == 14:
            self.start_pose = np.asarray(current_robot_pose, dtype=np.float64).copy()
        else:
            self.start_pose = STAND_POSE.copy()

    def step(self) -> Tuple[bool, np.ndarray]:
        """
        Advances one 50 Hz frame.
        Returns:
            (finished: bool, joint_targets: np.ndarray)
        """
        if not self.is_active or self.current_frame >= self.clip.num_frames:
            self.is_active = False
            return True, STAND_POSE.copy()

        target = self.projected_trajectory[self.current_frame].copy()

        # Inbound blending
        if self.current_frame < self.blend_frames:
            alpha = self.current_frame / float(self.blend_frames)
            target = (1.0 - alpha) * self.start_pose + alpha * target

        # Outbound blending
        frames_left = self.clip.num_frames - self.current_frame
        if frames_left <= self.blend_frames:
            beta = frames_left / float(self.blend_frames)
            target = (1.0 - beta) * STAND_POSE + beta * target

        self.current_frame += 1
        is_finished = self.current_frame >= self.clip.num_frames
        if is_finished:
            self.is_active = False

        return is_finished, target
