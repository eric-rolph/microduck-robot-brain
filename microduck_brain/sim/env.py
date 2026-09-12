"""Standalone MuJoCo simulation environment for Microduck biped.

Implements the standardized 61-D observation contract, 14-D action space,
BAM M6 actuator modeling, and mechanical backlash twin dynamics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import mujoco
import numpy as np

from microduck_brain.sim.backlash import BacklashManager
from microduck_brain.sim.bam_actuator import BamM6ActuatorModel, BamM6Config

# Default STAND2 pose (HOME_FRAME)
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


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by the inverse of quaternion [w, x, y, z]."""
    w = quat[0]
    xyz = quat[1:4]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


class MicroduckMuJoCoEnv:
    """MuJoCo simulation environment for Microduck matching the 61-D contract."""

    def __init__(
        self,
        xml_path: str | Path | None = None,
        use_backlash: bool = True,
        action_scale: float = 0.25,
        decimation: int = 10,
        bam_config: BamM6Config | None = None,
    ) -> None:
        self.action_scale = action_scale
        self.decimation = decimation

        if xml_path is None:
            base_dir = Path(__file__).parent / "mjcf"
            filename = "microduck_walk_backlash.xml" if use_backlash else "microduck_walk.xml"
            xml_path = base_dir / filename

        self.xml_path = str(xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Backlash twin manager
        self.backlash_mgr = BacklashManager(self.model)

        # BAM M6 actuator model
        self.bam = BamM6ActuatorModel(num_actuators=self.model.nu, config=bam_config)

        # Sensor and body IDs
        self.trunk_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.imu_gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.imu_accel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")

        # 13D command vector: twist(3), head_pose(4), body_pose(6)
        self.command = np.zeros(13, dtype=np.float32)
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.step_count = 0

        # Pre-allocate observation vector (61D)
        self.obs_dim = 61
        self.action_dim = self.model.nu

        # Geom classification for anti-fall ground collision gating
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        foot_names = {"left_foot_geom", "left_foot_sole", "right_foot_geom", "right_foot_sole"}
        self.foot_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in foot_names
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
        }
        self.non_foot_geom_ids = {
            i for i in range(self.model.ngeom)
            if i != self.floor_geom_id and i not in self.foot_geom_ids
        }
        self.fall_penalty = 50.0


    def set_command(
        self,
        lin_vel_x: float = 0.0,
        lin_vel_y: float = 0.0,
        ang_vel_z: float = 0.0,
        head_pose: np.ndarray | None = None,
        body_pose: np.ndarray | None = None,
    ) -> None:
        """Set the 13-D command vector for locomotion and posture."""
        self.command[0] = lin_vel_x
        self.command[1] = lin_vel_y
        self.command[2] = ang_vel_z

        if head_pose is not None:
            self.command[3:7] = head_pose[:4]
        else:
            self.command[3:7] = 0.0

        if body_pose is not None:
            self.command[7:13] = body_pose[:6]
        else:
            self.command[7:13] = 0.0

    def get_projected_gravity(self) -> np.ndarray:
        """Get projected gravity vector [gx, gy, gz] in trunk base frame."""
        quat = self.data.xquat[self.trunk_body_id].copy().astype(np.float32)
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return quat_rotate_inverse(quat, world_gravity)

    def get_base_angular_velocity(self) -> np.ndarray:
        """Get base angular velocity from IMU gyro sensor."""
        sensor_adr = self.model.sensor_adr[self.imu_gyro_id]
        return self.data.sensordata[sensor_adr : sensor_adr + 3].copy().astype(np.float32)

    def get_base_linear_velocity(self) -> np.ndarray:
        """Get trunk base linear velocity in world frame."""
        # 3D linear velocity is the first 3 DOF of the root freejoint
        return self.data.qvel[0:3].copy().astype(np.float32)

    def get_observation(self) -> np.ndarray:
        """Construct the standardized 61-D observation vector.

        Components:
        1. base_ang_vel: 3D
        2. projected_gravity: 3D
        3. joint_pos relative to DEFAULT_POSE: 14D (through backlash)
        4. joint_vel: 14D (through backlash)
        5. last_action: 14D
        6. command: 13D
        Total: 3 + 3 + 14 + 14 + 14 + 13 = 61D
        """
        ang_vel = self.get_base_angular_velocity()
        proj_grav = self.get_projected_gravity()

        q_enc = self.backlash_mgr.read_encoder_positions(self.data)
        v_enc = self.backlash_mgr.read_encoder_velocities(self.data)

        q_rel = q_enc - DEFAULT_POSE[: self.model.nu]

        obs = np.concatenate(
            [ang_vel, proj_grav, q_rel, v_enc, self.last_action, self.command]
        ).astype(np.float32)
        return obs

    def reset(self, randomize_noise: float = 0.01) -> np.ndarray:
        """Reset environment to standing keyframe pose."""
        mujoco.mj_resetData(self.model, self.data)

        # Apply STAND2 keyframe if present
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "STAND2")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            # Fallback manual default pose
            self.data.qpos[0:3] = [0.0, 0.0, 0.14]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            # Set joint positions
            for i, adr in enumerate(self.backlash_mgr.servo_qpos_adr):
                self.data.qpos[adr] = DEFAULT_POSE[i]
            self.data.ctrl[:] = DEFAULT_POSE[: self.model.nu]

        if randomize_noise > 0.0:
            noise = np.random.uniform(
                -randomize_noise, randomize_noise, size=len(self.backlash_mgr.servo_qpos_adr)
            )
            for i, adr in enumerate(self.backlash_mgr.servo_qpos_adr):
                self.data.qpos[adr] += noise[i]

        mujoco.mj_forward(self.model, self.data)

        # Reset BAM M6 internal state
        self.bam.reset(initial_targets=DEFAULT_POSE[: self.model.nu])
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.step_count = 0

        return self.get_observation()

    def check_non_foot_ground_collision(self) -> bool:
        """Returns True if any non-foot body collides with the ground floor."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.floor_geom_id:
                if g2 in self.non_foot_geom_ids:
                    return True
            elif g2 == self.floor_geom_id:
                if g1 in self.non_foot_geom_ids:
                    return True
        return False

    def is_terminated(self) -> bool:
        """Check for fall, ground strike, or height collapse."""
        trunk_z = min(float(self.data.qpos[2]), float(self.data.xpos[self.trunk_body_id][2]))
        # Standing height is ~0.14 m. Drop below 0.080 m is a collapse.
        if trunk_z < 0.080:
            return True

        proj_grav = self.get_projected_gravity()
        # If gravity z > -0.75, robot has tilted > 41 degrees (unrecoverable fall)
        if proj_grav[2] > -0.75:
            return True

        # Check for non-foot collision (knees, head, trunk hitting floor)
        if self.check_non_foot_ground_collision():
            return True

        return False

    def compute_reward(self, obs: np.ndarray, action: np.ndarray, is_fall: bool = False) -> float:
        """Compute tracking, posture, balance survival, and regularization rewards."""
        if is_fall:
            return -self.fall_penalty

        # 1. Linear velocity tracking in robot heading
        lin_vel_world = self.get_base_linear_velocity()
        quat = self.data.xquat[self.trunk_body_id].copy().astype(np.float32)
        lin_vel_body = quat_rotate_inverse(quat, lin_vel_world)

        vx_target = self.command[0]
        vy_target = self.command[1]
        vel_error = (lin_vel_body[0] - vx_target) ** 2 + (lin_vel_body[1] - vy_target) ** 2
        r_linvel = math.exp(-4.0 * vel_error)

        # 2. Yaw velocity tracking
        ang_vel = obs[0:3]
        wz_target = self.command[2]
        ang_error = (ang_vel[2] - wz_target) ** 2
        r_angvel = math.exp(-3.0 * ang_error)

        # 3. Upright orientation: projected gravity z should be near -1.0
        proj_grav = obs[3:6]
        tilt_error = float(proj_grav[0] ** 2 + proj_grav[1] ** 2)
        r_upright = math.exp(-4.0 * tilt_error)

        # Forward velocity progress term: breaks the zero-velocity posture freeze
        forward_vel = float(lin_vel_body[0])
        if vx_target > 0.01:
            r_progress = 3.0 * float(np.clip(forward_vel / vx_target, -0.5, 1.2))
            r_survival = 2.0 * r_upright * max(0.2, min(1.0, forward_vel / vx_target))
        else:
            r_progress = 0.0
            r_survival = 2.0 * r_upright

        # 5. Height maintenance: trunk z ~ 0.14 m
        trunk_z = float(self.data.xpos[self.trunk_body_id][2])
        r_height = math.exp(-50.0 * ((trunk_z - 0.14) ** 2))

        # 6. Anti-fall stabilization penalties
        # Lateral drift penalty: penalize uncommanded sideways sliding
        side_drift_penalty = float(lin_vel_body[1] ** 2)
        # Roll and pitch angular velocity penalty: penalize trunk rocking/wobbling
        wobble_penalty = float(ang_vel[0] ** 2 + ang_vel[1] ** 2)

        # 7. Smoothness penalties
        action_diff = float(np.sum((action - self.last_action) ** 2))
        torque_penalty = float(np.sum(self.bam.last_torques ** 2))

        total_reward = (
            1.5 * r_linvel
            + r_progress
            + 1.0 * r_angvel
            + 1.0 * r_upright
            + r_survival
            + 1.0 * r_height
            - 0.5 * side_drift_penalty
            - 0.15 * wobble_penalty
            - 0.05 * action_diff
            - 0.005 * torque_penalty
        )
        return float(total_reward)

    def apply_push_disturbance(self, delta_vx: float, delta_vy: float) -> None:
        """Inject an impulsive velocity perturbation to the trunk base to train recovery."""
        self.data.qvel[0] += float(delta_vx)
        self.data.qvel[1] += float(delta_vy)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by one policy step (decimated physics sub-steps).

        Parameters
        ----------
        action : np.ndarray
            Policy action delta (14D) in [-1.0, 1.0].

        Returns
        -------
        obs : np.ndarray
            61-D observation vector.
        reward : float
            Step reward.
        terminated : bool
            True if robot fell.
        truncated : bool
            False (or max steps exceeded).
        info : dict
            Diagnostic metrics (battery voltage, torque norms, velocities).
        """
        clipped_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_positions = DEFAULT_POSE[: self.model.nu] + clipped_action * self.action_scale

        # Advance BAM M6 transport delay queue once per policy step (50 Hz / 20 ms)
        delayed_targets = self.bam.step_delay(target_positions)

        # Run decimated sub-steps
        for _ in range(self.decimation):
            q_enc = self.backlash_mgr.read_encoder_positions(self.data)
            v_enc = self.backlash_mgr.read_encoder_velocities(self.data)

            # BAM M6 computes motor torques & updates dynamic voltage/back-EMF limits
            torques = self.bam.compute_torques(delayed_targets, q_enc, v_enc, advance_delay=False)

            # Actuator force coupling: enforce BAM dynamic torque limits on MuJoCo solver
            dynamic_limits = self.bam.last_torque_limits
            self.model.actuator_forcerange[:, 0] = -dynamic_limits
            self.model.actuator_forcerange[:, 1] = dynamic_limits

            # Apply setpoints
            self.data.ctrl[:] = delayed_targets

            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self.get_observation()
        terminated = self.is_terminated()
        reward = self.compute_reward(obs, clipped_action, is_fall=terminated)
        truncated = self.step_count >= 1000

        info = {
            "battery_voltage": self.bam.battery_voltage,
            "battery_current": self.bam.last_current,
            "trunk_height": float(self.data.xpos[self.trunk_body_id][2]),
            "projected_gravity": obs[3:6],
            "step": self.step_count,
            "is_fall": terminated,
        }

        self.last_action = clipped_action.copy()
        return obs, reward, terminated, truncated, info

