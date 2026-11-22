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

from microduck_brain.reactor_onnx_compiler import OFFICIAL_DEFAULT_POSE
from microduck_brain.sim.backlash import BacklashManager
from microduck_brain.sim.bam_actuator import BamM6ActuatorModel, BamM6Config
from microduck_brain.sim.exact_bam_plant import (
    BAM_STIFF_SOLIMP_FRICTION,
    BAM_STIFF_SOLREF_FRICTION,
    XL330_M6_PARAMS,
)

# Default STAND2 pose (HOME_FRAME), exact 7-decimal official contract
DEFAULT_POSE = OFFICIAL_DEFAULT_POSE.copy()


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
        action_scale: float = 1.0,
        decimation: int = 4,
        bam_config: BamM6Config | None = None,
        use_exact_bam: bool = False,
    ) -> None:
        self.action_scale = action_scale
        self.decimation = decimation
        self.use_exact_bam = bool(use_exact_bam)

        if xml_path is None:
            base_dir = Path(__file__).parent / "mjcf"
            filename = "microduck_walk_backlash.xml" if use_backlash else "microduck_walk.xml"
            xml_path = base_dir / filename

        self.xml_path = str(xml_path)
        if self.use_exact_bam:
            p = XL330_M6_PARAMS
            force_limit = p["vin"] * p["kt"] / p["R"]
            spec = mujoco.MjSpec.from_file(self.xml_path)
            for act in spec.actuators:
                tgt = act.target
                tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
                if tgt_name.startswith("passive_"):
                    continue
                act.set_to_motor()
                act.forcelimited = True
                act.forcerange = (-force_limit, force_limit)
                act.ctrllimited = False
                act.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                for joint in spec.joints:
                    if joint.name == tgt_name:
                        joint.damping = np.zeros((3, 1))
                        joint.frictionloss = 0.0
                        joint.solref_friction = BAM_STIFF_SOLREF_FRICTION
                        joint.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
                        break
            self.model = spec.compile()
        else:
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)

        # Backlash twin manager
        self.backlash_mgr = BacklashManager(self.model)
        self._servo_dof_adr = np.array(self.backlash_mgr.servo_qvel_adr, dtype=np.int64)
        self._servo_qpos_adr = np.array(self.backlash_mgr.servo_qpos_adr, dtype=np.int64)
        self._servo_joint_idx = np.array(
            [int(self.model.actuator_trnid[i, 0]) for i in range(self.model.nu)], dtype=np.int64
        )
        self._exact_lookup = np.full(int(max(self.model.njnt, self.model.nv)) + 1, -1)
        self._exact_lookup[self._servo_joint_idx] = np.arange(self.model.nu)
        self._friction_dof_type = mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF.value
        if self.use_exact_bam:
            self.model.dof_armature[self._servo_dof_adr] = XL330_M6_PARAMS["armature"]
            mujoco.mj_setConst(self.model, self.data)

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
        foot_body_names = {"ankle_left", "ankle_right", "left_foot", "right_foot"}
        foot_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in foot_body_names
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
        }
        self.foot_geom_ids = set()
        for i in range(self.model.ngeom):
            g_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            b_id = self.model.geom_bodyid[i]
            if b_id in foot_body_ids:
                self.foot_geom_ids.add(i)
            elif g_name and any(k in g_name.lower() for k in ("foot", "sole", "ankle")):
                self.foot_geom_ids.add(i)

        # Collect all robot bodies (descendants of trunk_base)
        robot_body_ids = {self.trunk_body_id}
        for b in range(self.model.nbody):
            curr = b
            while curr > 0:
                if curr == self.trunk_body_id:
                    robot_body_ids.add(b)
                    break
                curr = self.model.body_parentid[curr]

        self.robot_body_ids = robot_body_ids

        # Classify mouth / jaw geoms for intentional ground retrieval contact
        mouth_body_names = {"jaw_soft", "beak_jaw", "beak_lower"}
        mouth_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in mouth_body_names
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
        }
        self.mouth_geom_ids = set()
        for i in range(self.model.ngeom):
            b_id = self.model.geom_bodyid[i]
            g_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if b_id in mouth_body_ids or (g_name and any(k in g_name.lower() for k in ("beak", "jaw", "mouth"))):
                self.mouth_geom_ids.add(i)

        # Only robot bodies that are neither floor, feet, nor world obstacles
        self.non_foot_geom_ids = {
            i for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] in self.robot_body_ids
            and i != self.floor_geom_id
            and i not in self.foot_geom_ids
        }
        self.allow_mouth_contact = False
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

        if self.use_exact_bam:
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
            self.model.dof_frictionloss[self._servo_dof_adr] = 0.0
            self.model.dof_damping[self._servo_dof_adr] = 0.0
            mujoco.mj_forward(self.model, self.data)
        else:
            mujoco.mj_forward(self.model, self.data)
            # Settle robot onto ground plane in nominal STAND pose
            for _ in range(20):
                mujoco.mj_step(self.model, self.data)

        # Reset BAM M6 internal state
        self.bam.reset(initial_targets=DEFAULT_POSE[: self.model.nu])
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.step_count = 0

        return self.get_observation()

    def check_non_foot_ground_collision(self) -> bool:
        """Returns True if any non-foot robot body collides with the ground floor."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.floor_geom_id:
                if g2 in self.non_foot_geom_ids:
                    if self.allow_mouth_contact and g2 in self.mouth_geom_ids:
                        continue
                    return True
            elif g2 == self.floor_geom_id:
                if g1 in self.non_foot_geom_ids:
                    if self.allow_mouth_contact and g1 in self.mouth_geom_ids:
                        continue
                    return True
        return False

    def is_terminated(self) -> bool:
        """Check for fall, ground strike, or height collapse."""
        trunk_z = min(float(self.data.qpos[2]), float(self.data.xpos[self.trunk_body_id][2]))
        # Standing height is ~0.12 m. Drop below 0.060 m (or 0.045 m during intentional crouch) is a collapse.
        min_trunk_z = 0.045 if self.allow_mouth_contact else 0.060
        if trunk_z < min_trunk_z:
            return True

        proj_grav = self.get_projected_gravity()
        # If gravity z > -0.50, robot has tilted > 60 degrees (unrecoverable fall)
        # Note: during intentional ground retrieval and carrying (allow_mouth_contact=True),
        # forward bow pitch up to ~104 degrees (proj_grav[2] up to 0.25) is permitted for deep floor reach
        tilt_limit = 0.25 if self.allow_mouth_contact else -0.50
        if proj_grav[2] > tilt_limit:
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
            if self.use_exact_bam:
                p = XL330_M6_PARAMS
                q_s = self.data.qpos[self._servo_qpos_adr]
                dq_s = self.data.qvel[self._servo_dof_adr]
                duty = np.clip((delayed_targets - q_s) * p["kp_fw"] * p["error_gain"], -p["max_pwm"], p["max_pwm"])
                volts = p["vin"] * duty
                torque = p["kt"] * volts / p["R"] - (p["kt"] ** 2) * dq_s / p["R"]
                self.data.ctrl[:] = torque
                self.bam.last_torques = torque.astype(np.float32)
                self.bam.update_battery_voltage(torque)

                te = -self.data.qfrc_bias[self._servo_dof_adr] + self.data.qfrc_constraint[self._servo_dof_adr]
                ff = np.zeros(self.model.nu, dtype=np.float64)
                if len(self.data.efc_id):
                    rows = np.flatnonzero(self.data.efc_type == self._friction_dof_type)
                    if len(rows):
                        ids = self.data.efc_id[rows]
                        ok = ids < len(self._exact_lookup)
                        which = self._exact_lookup[ids[ok]]
                        sel = which >= 0
                        np.add.at(ff, which[sel], self.data.efc_force[rows[ok][sel]])
                te -= ff
                mt = self.data.qfrc_actuator[self._servo_dof_adr]
                gt = np.abs(te * p["load_friction_external"] - mt * p["load_friction_motor"])
                gts = np.abs(te * p["load_friction_external_stribeck"] - mt * p["load_friction_motor_stribeck"])
                sc = np.exp(-(np.abs(dq_s / p["dtheta_stribeck"]) ** p["alpha"]))
                fl = p["friction_base"] + gt + sc * p["friction_stribeck"] + gts * sc
                eq = np.sign(te) != np.sign(mt)
                g2m = p["load_friction_external_quad"] * (np.abs(te) ** 2)
                g2e = p["load_friction_motor_quad"] * (np.abs(mt) ** 2)
                fl += sc * ((np.abs(te) < np.abs(mt)) * g2m + (np.abs(te) > np.abs(mt)) * g2e) * eq
                self.model.dof_frictionloss[self._servo_dof_adr] = fl
                self.model.dof_damping[self._servo_dof_adr] = p["friction_viscous"]
            else:
                q_enc = self.backlash_mgr.read_encoder_positions(self.data)
                v_enc = self.backlash_mgr.read_encoder_velocities(self.data)

                # BAM M6 computes motor torques, battery sag, and dynamic back-EMF limits
                self.bam.compute_torques(delayed_targets, q_enc, v_enc, advance_delay=False)

                # Couple BAM dynamic voltage-sag / back-EMF torque limits into MuJoCo's
                # position actuators while preserving the XML's calibrated chosen_actuator
                # back-EMF damping (0.053) and baseline frictionloss (0.0048). For full
                # Stribeck + load-dependent XL330 M6 torque dynamics, set use_exact_bam=True.
                dynamic_limits = self.bam.last_torque_limits
                self.model.actuator_forcerange[:, 0] = -dynamic_limits
                self.model.actuator_forcerange[:, 1] = dynamic_limits

                # Apply position setpoints to position actuators
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

