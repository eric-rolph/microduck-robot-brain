"""Mechanical backlash twin dynamics and encoder reading through backlash play.

On the physical Microduck biped, magnetic encoders read joint angles on the horn
output side, after 3D-printed tolerances and gearbox backlash play (+/-1 degree).
The simulation passes q_enc = q_servo + q_backlash to match physical observations.
"""

from __future__ import annotations

import mujoco
import numpy as np


class BacklashManager:
    """Manages joint mapping and encoder feedback through backlash hinges."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.nu = model.nu

        # Actuated servo joint qpos and qvel addresses
        self.servo_qpos_adr: list[int] = []
        self.servo_qvel_adr: list[int] = []

        # Associated passive backlash joint qpos and qvel addresses (or None)
        self.backlash_qpos_adr: list[int | None] = []
        self.backlash_qvel_adr: list[int | None] = []
        self.has_backlash: bool = False

        self._build_joint_maps()

    def _build_joint_maps(self) -> None:
        """Resolve joint addresses for actuated servos and backlash hinges."""
        for act_idx in range(self.nu):
            # Actuator transmission targets joint ID
            joint_id = int(self.model.actuator_trnid[act_idx, 0])
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            qvel_adr = int(self.model.jnt_dofadr[joint_id])

            self.servo_qpos_adr.append(qpos_adr)
            self.servo_qvel_adr.append(qvel_adr)

            # Search for matching passive backlash joint: 'passive_<joint>_backlash'
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            bl_name = f"passive_{joint_name}_backlash"
            bl_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, bl_name)

            if bl_joint_id >= 0:
                self.has_backlash = True
                self.backlash_qpos_adr.append(int(self.model.jnt_qposadr[bl_joint_id]))
                self.backlash_qvel_adr.append(int(self.model.jnt_dofadr[bl_joint_id]))
            else:
                self.backlash_qpos_adr.append(None)
                self.backlash_qvel_adr.append(None)

    def read_encoder_positions(self, data: mujoco.MjData) -> np.ndarray:
        """Read joint positions through backlash deadbands (14D)."""
        positions = np.zeros(self.nu, dtype=np.float32)
        for i in range(self.nu):
            q_servo = float(data.qpos[self.servo_qpos_adr[i]])
            bl_adr = self.backlash_qpos_adr[i]
            q_bl = float(data.qpos[bl_adr]) if bl_adr is not None else 0.0
            positions[i] = q_servo + q_bl
        return positions

    def read_encoder_velocities(self, data: mujoco.MjData) -> np.ndarray:
        """Read joint velocities through backlash deadbands (14D)."""
        velocities = np.zeros(self.nu, dtype=np.float32)
        for i in range(self.nu):
            v_servo = float(data.qvel[self.servo_qvel_adr[i]])
            bl_adr = self.backlash_qvel_adr[i]
            v_bl = float(data.qvel[bl_adr]) if bl_adr is not None else 0.0
            velocities[i] = v_servo + v_bl
        return velocities
