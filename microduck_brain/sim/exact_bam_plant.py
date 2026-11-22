"""Exact XL330 M6 BAM MuJoCo Plant and Recovery Supervisor for ReActor 61-D Skills.

Implements the exact voltage-controlled Dynamixel XL330 M6 actuator model
(``FastBamUpdate`` from ``microduck-reactor``) on ``microduck_walk.xml`` without
requiring external ``bam`` or POSIX-only ``termios`` dependencies.

Features:
  * Exact XL330 M6 electrical & Stribeck/load-dependent quadratic gearbox friction parameters.
  * 1-control (20 ms) transport delay and 1-control ``joint_vel`` observation lag.
  * Support for P1 / P1x1.5 domain randomization (foot friction, servo Kp, viscous damping,
    sensor noise) and external horizontal trunk push disturbances.
  * Closed-loop execution of self-contained ``61 -> 14`` ``official61`` ReActor ONNX skills,
    seamless handover to ``BEST_alpha_walking.onnx``, and automatic fall recovery via
    ``BEST_alpha_stand.onnx``.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort

from microduck_brain.reactor_onnx_compiler import (
    OFFICIAL_DEFAULT_POSE,
    OFFICIAL_JOINT_NAMES,
    PHASE_SCHEMA_FOURIER_V1,
    compute_phase_command,
    read_reactor_onnx_metadata,
)

_COMPILED_MODEL_CACHE: dict[tuple[str, float], mujoco.MjModel] = {}

PHYSICS_DT_S = 0.005
CONTROL_DT_S = 0.02
STEPS_PER_CONTROL = 4
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")
FOOT_SITES = ("left_foot", "right_foot")

# Exact XL330 M6 parameters from bam.model.load_model('xl330', 'm6') + microduck_constants.py
XL330_M6_PARAMS: dict[str, float] = {
    "kt": 0.36601349688984386,
    "R": 2.8113923539223227,
    "armature": 0.0018077432831600838,
    "kp_fw": 200.0,
    "error_gain": 0.0028773775022263564,
    "max_pwm": 1.0,
    "vin": 7.4,
    "vin_min": 6.0,
    "friction_base": 0.004771183165566,
    "friction_stribeck": 0.004676345799486616,
    "load_friction_motor": 0.2667860954283698,
    "load_friction_external": 8.515871897059342e-06,
    "load_friction_motor_stribeck": 1.0722918395099123e-05,
    "load_friction_external_stribeck": 0.08077928978935671,
    "load_friction_motor_quad": 0.009972471242139415,
    "load_friction_external_quad": 0.004902565732332559,
    "dtheta_stribeck": 2.890372094130307,
    "alpha": 8.683259907618984,
    "friction_viscous": 0.005359668274599504,
}

BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


@dataclass
class PlantRandomization:
    """Domain randomization & sensor noise matching microduck-reactor P1 / P1x1.5."""

    friction_scale: float = 1.0
    kp_scale: float = 1.0
    viscous_scale: float = 1.0
    sensor_noise_scale: float = 0.0
    gyro_noise_rad_s: float = 0.05
    gravity_noise: float = 0.02
    qpos_noise_rad: float = 0.01
    qvel_noise_rad_s: float = 0.15

    @classmethod
    def draw_p1(cls, rng: np.random.Generator, scale: float = 1.0, noise: bool = True) -> PlantRandomization:
        return cls(
            friction_scale=float(rng.uniform(1.0 - 0.30 * scale, 1.0 + 0.30 * scale)),
            kp_scale=float(rng.uniform(1.0 - 0.15 * scale, 1.0 + 0.15 * scale)),
            viscous_scale=float(rng.uniform(1.0 - 0.15 * scale, 1.0 + 0.15 * scale)),
            sensor_noise_scale=scale if noise else 0.0,
        )

    def sample_sensor_noise(
        self, rng: np.random.Generator | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if rng is None or self.sensor_noise_scale <= 0.0:
            return (
                np.zeros(3, dtype=np.float64),
                np.zeros(14, dtype=np.float64),
                np.zeros(14, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
        s = self.sensor_noise_scale
        ng = rng.uniform(-self.gyro_noise_rad_s * s, self.gyro_noise_rad_s * s, size=3)
        nq = rng.uniform(-self.qpos_noise_rad * s, self.qpos_noise_rad * s, size=14)
        nv = rng.uniform(-self.qvel_noise_rad_s * s, self.qvel_noise_rad_s * s, size=14)
        ngr = rng.uniform(-self.gravity_noise * s, self.gravity_noise * s, size=3)
        return ng, nq, nv, ngr


class ExactBamPlant:
    """Dependency-free exact XL330 M6 BAM MuJoCo plant for 61-D official/ReActor policies."""

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        vin: float = XL330_M6_PARAMS["vin"],
        randomization: PlantRandomization | None = None,
        seed: int = 0,
    ) -> None:
        if xml_path is None:
            xml_path = Path(__file__).parent / "mjcf" / "microduck_walk.xml"
        self.xml_path = Path(xml_path).resolve()
        self.vin = float(vin)
        self.randomization = randomization or PlantRandomization()
        self.rng = np.random.default_rng(seed)

        self.p = dict(XL330_M6_PARAMS)
        self.kt = self.p["kt"]
        self.R = self.p["R"]
        self.kp = self.p["kp_fw"] * self.randomization.kp_scale
        self.error_gain = self.p["error_gain"]
        self.max_pwm = self.p["max_pwm"]
        self.p["friction_viscous"] = XL330_M6_PARAMS["friction_viscous"] * self.randomization.viscous_scale

        force_limit = self.vin * self.kt / self.R
        cache_key = (str(self.xml_path), round(force_limit, 9))
        cached = _COMPILED_MODEL_CACHE.get(cache_key)
        if cached is not None:
            self.model = copy.copy(cached)
        else:
            spec = mujoco.MjSpec.from_file(str(self.xml_path))
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

            compiled = spec.compile()
            compiled.opt.timestep = PHYSICS_DT_S
            _COMPILED_MODEL_CACHE[cache_key] = compiled
            self.model = copy.copy(compiled)
        self.data = mujoco.MjData(self.model)

        jnames = tuple(self.model.joint(int(self.model.actuator_trnid[i, 0])).name for i in range(self.model.nu))
        if jnames != OFFICIAL_JOINT_NAMES:
            raise ValueError(f"Actuator joint order mismatch: {jnames}")

        if self.randomization.friction_scale != 1.0:
            for gname in FOOT_GEOMS:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
                if gid >= 0:
                    self.model.geom_friction[gid, 0] *= self.randomization.friction_scale

        self.qadr = np.array([self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]] for i in range(self.model.nu)])
        self.dadr = np.array([self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]] for i in range(self.model.nu)])
        self.joint_idx = np.array([self.model.actuator_trnid[i, 0] for i in range(self.model.nu)])
        self.model.dof_armature[self.dadr] = self.p["armature"]
        mujoco.mj_setConst(self.model, self.data)

        self.trunk = self.model.body("trunk_base").id
        self.gyro_adr = int(self.model.sensor_adr[self.model.sensor("imu_ang_vel").id])
        self.foot_sites = np.array(
            [self.model.site(n).id for n in FOOT_SITES if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n) >= 0]
        )

        size = int(max(self.model.njnt, self.model.nv)) + 1
        self.lookup = np.full(size, -1)
        self.lookup[self.joint_idx] = np.arange(len(self.joint_idx))
        self.friction_dof = mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF.value

        self.stand_key = self.model.key("STAND").id
        self.pending = OFFICIAL_DEFAULT_POSE.astype(np.float64).copy()
        self.prev_vel = np.zeros(14, dtype=np.float32)
        self.last_action = np.zeros(14, dtype=np.float32)
        self.reset_stand()

    def apply_randomization(self, randomization: PlantRandomization) -> None:
        """UpdateKp/viscous/friction scales in-place without recompiling MjSpec."""
        # Restore base foot friction before applying new scale
        ratio = randomization.friction_scale / max(self.randomization.friction_scale, 1e-8)
        for gname in FOOT_GEOMS:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                self.model.geom_friction[gid, 0] *= ratio
        self.randomization = randomization
        self.kp = XL330_M6_PARAMS["kp_fw"] * randomization.kp_scale
        self.p["friction_viscous"] = XL330_M6_PARAMS["friction_viscous"] * randomization.viscous_scale

    def reset_stand(self, qpos: np.ndarray | None = None) -> np.ndarray:
        """Reset plant to the STAND keyframe with 1-control command delay initialized."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key)
        if qpos is not None:
            self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.model.dof_frictionloss[self.dadr] = 0.0
        self.model.dof_damping[self.dadr] = 0.0
        self.pending = self.data.qpos[self.qadr].copy()
        self.prev_vel = np.zeros(14, dtype=np.float32)
        self.last_action = np.zeros(14, dtype=np.float32)
        return self.observe(np.zeros(13, dtype=np.float32), lag_velocity=True)

    def projected_gravity(self) -> np.ndarray:
        return (-self.data.xmat[self.trunk].reshape(3, 3)[2]).copy()

    def is_fallen(self) -> bool:
        g = self.projected_gravity()
        return bool(self.data.qpos[2] < 0.060 or g[2] > -0.50)

    def observe(
        self,
        command_13: np.ndarray,
        *,
        lag_velocity: bool = True,
        last_action: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build the 61-D observation vector."""
        ng, nq, nv, ngr = self.randomization.sample_sensor_noise(self.rng)
        gyro = self.data.sensordata[self.gyro_adr : self.gyro_adr + 3] + ng
        grav = self.projected_gravity() + ngr
        q_rel = (self.data.qpos[self.qadr] + nq) - OFFICIAL_DEFAULT_POSE.astype(np.float64)
        cur_vel = (self.data.qvel[self.dadr] + nv).astype(np.float32)
        vel_obs = self.prev_vel if lag_velocity else cur_vel
        self.prev_vel = cur_vel.copy()
        act_hist = self.last_action if last_action is None else np.asarray(last_action, dtype=np.float32)
        cmd = np.asarray(command_13, dtype=np.float32).reshape(13)
        return np.concatenate([gyro, grav, q_rel, vel_obs, act_hist, cmd]).astype(np.float32)

    def step_control(self, target_q: np.ndarray) -> None:
        """Advance 1 control interval (20 ms = 4 physics steps) with 1-control command delay."""
        q_target = self.pending.copy()
        self.pending = np.asarray(target_q, dtype=np.float64).reshape(14).copy()
        p = self.p
        for _ in range(STEPS_PER_CONTROL):
            q = self.data.qpos[self.qadr]
            dq = self.data.qvel[self.dadr]
            duty_cycle = np.clip((q_target - q) * self.kp * self.error_gain, -self.max_pwm, self.max_pwm)
            volts = self.vin * duty_cycle
            torque = self.kt * volts / self.R - (self.kt**2) * dq / self.R
            self.data.ctrl[:] = torque

            te = -self.data.qfrc_bias[self.dadr] + self.data.qfrc_constraint[self.dadr]
            ff = np.zeros(14, dtype=np.float64)
            if len(self.data.efc_id):
                rows = np.flatnonzero(self.data.efc_type == self.friction_dof)
                if len(rows):
                    ids = self.data.efc_id[rows]
                    ok = ids < len(self.lookup)
                    which = self.lookup[ids[ok]]
                    sel = which >= 0
                    np.add.at(ff, which[sel], self.data.efc_force[rows[ok][sel]])
            te -= ff
            mt = self.data.qfrc_actuator[self.dadr]
            et = te
            gt = np.abs(et * p["load_friction_external"] - mt * p["load_friction_motor"])
            gts = np.abs(et * p["load_friction_external_stribeck"] - mt * p["load_friction_motor_stribeck"])
            sc = np.exp(-(np.abs(dq / p["dtheta_stribeck"]) ** p["alpha"]))
            fl = p["friction_base"] + gt + sc * p["friction_stribeck"] + gts * sc
            eq = np.sign(et) != np.sign(mt)
            dm = np.abs(et) < np.abs(mt)
            de = np.abs(et) > np.abs(mt)
            g2m = p["load_friction_external_quad"] * (np.abs(et) ** 2)
            g2e = p["load_friction_motor_quad"] * (np.abs(mt) ** 2)
            fl += sc * (dm * g2m + de * g2e) * eq
            self.model.dof_frictionloss[self.dadr] = fl
            self.model.dof_damping[self.dadr] = p["friction_viscous"]
            mujoco.mj_step(self.model, self.data)

    def step_action(self, action_14: np.ndarray) -> None:
        """Apply a 14-D default-pose-relative action."""
        act = np.asarray(action_14, dtype=np.float32).reshape(14)
        self.last_action = act.copy()
        self.step_control(OFFICIAL_DEFAULT_POSE.astype(np.float64) + act.astype(np.float64))
