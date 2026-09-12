"""
Mocap playback and retargeting engine for Microduck 14-DOF biped.
Replays motions translated from human motion capture datasets with
stability-first projection preserving the biped support polygon.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


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


class MocapPlayer:
    """
    50 Hz Mocap trajectory player with stability-first projection.
    Streams joint setpoints frame-by-frame, ensuring smooth blend from/to standing.
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
