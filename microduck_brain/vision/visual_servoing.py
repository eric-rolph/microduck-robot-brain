"""Visual Servoing Controller for Microduck.

Translates visual track bearing and range into closed-loop velocity commands (vx, wz)
for the neural walking policy.
"""

from __future__ import annotations

import numpy as np
from microduck_brain.vision.tracker import VisualTrackState


class VisualServoingController:
    """Proportional visual servoing controller toward a locked target."""

    def __init__(
        self,
        kp_yaw: float = 0.75,
        kp_dist: float = 0.30,
        target_range_m: float = 0.18,
        max_vx: float = 0.08,
        max_wz: float = 0.18,
    ) -> None:
        self.kp_yaw = kp_yaw
        self.kp_dist = kp_dist
        self.target_range_m = target_range_m
        self.max_vx = max_vx
        self.max_wz = max_wz

    def compute_commands(
        self,
        track_state: VisualTrackState,
    ) -> tuple[float, float, bool]:
        """Compute (cmd_vx, cmd_wz, target_reached) from current visual track state."""
        if not track_state.target_locked:
            return 0.0, 0.0, False

        bearing = track_state.bearing_rad
        dist = track_state.range_m

        # Steer toward target bearing (negative wz turns right in robot frame)
        wz = float(np.clip(-self.kp_yaw * bearing, -self.max_wz, self.max_wz))

        # If pointing too far away (> 25 deg), turn in place first
        if abs(bearing) > 0.44:
            return 0.0, wz, False

        # Forward velocity proportional to distance remaining
        dist_err = dist - self.target_range_m
        if dist_err <= 0.02 and abs(bearing) < 0.12:
            return 0.0, 0.0, True

        vx = float(np.clip(self.kp_dist * dist_err, 0.0, self.max_vx))
        return vx, wz, False
