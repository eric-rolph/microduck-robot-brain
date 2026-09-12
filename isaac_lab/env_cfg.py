"""
Isaac Lab environment configuration for Microduck quadruped locomotion.
Sets up ManagerBasedRLEnvCfg with base CoM randomization,
neck impulse disturbance wrenches, and attitude velocity command sampling.
"""

from __future__ import annotations
import math
import torch
from dataclasses import MISSING

try:
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.managers import (
        CommandTerm,
        CommandTermCfg,
        EventTermCfg,
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        SceneEntityCfg,
    )
    import isaaclab.envs.mdp as mdp
    from isaaclab.utils import configclass
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    ISAAC_LAB_AVAILABLE = False
    # Mock base types for standalone validation and linting outside Omniverse
    class Articulation:  # type: ignore[no-redef]
        data: Any

    class CommandTerm:  # type: ignore[no-redef]
        def __init__(self, cfg: Any, env: Any):
            self.cfg = cfg
            self.env = env
            self.num_envs = getattr(env, "num_envs", 1)
            self.device = getattr(env, "device", "cpu")

    def configclass(cls):  # type: ignore[no-redef]
        return cls

    class CommandTermCfg: pass
    class EventTermCfg:
        def __init__(self, **kwargs): pass
    class ObservationGroupCfg: pass
    class ObservationTermCfg:
        def __init__(self, **kwargs): pass
    class RewardTermCfg:
        def __init__(self, **kwargs): pass
    class SceneEntityCfg:
        def __init__(self, name: str = "", **kwargs):
            self.name = name
    class ManagerBasedRLEnvCfg: pass
    class ArticulationCfg: pass
    class mdp:  # type: ignore[no-redef]
        @staticmethod
        def randomize_rigid_body_com(**kwargs): pass
        @staticmethod
        def apply_external_force_torque(**kwargs): pass
        @staticmethod
        def randomize_rigid_body_mass(**kwargs): pass
        @staticmethod
        def projected_gravity(**kwargs): pass
        @staticmethod
        def base_lin_vel(**kwargs): pass
        @staticmethod
        def base_ang_vel(**kwargs): pass
        @staticmethod
        def joint_pos_rel(**kwargs): pass
        @staticmethod
        def joint_vel_rel(**kwargs): pass
        @staticmethod
        def last_action(**kwargs): pass
        @staticmethod
        def generated_commands(**kwargs): pass
        @staticmethod
        def track_lin_vel_xy_exp(**kwargs): pass
        @staticmethod
        def track_ang_vel_z_exp(**kwargs): pass
        @staticmethod
        def joint_torques_l2(**kwargs): pass
        @staticmethod
        def action_rate_l2(**kwargs): pass


# ==============================================================================
# 1. Custom Command Generator: Velocity + Commanded Attitude
# ==============================================================================

class AttitudeVelocityCommand(CommandTerm):
    """
    Samples linear/angular velocities (vx, vy, wz) alongside target
    trunk roll and pitch offsets relative to the gravity vector.
    """
    cfg: "AttitudeVelocityCommandCfg"

    def __init__(self, cfg: "AttitudeVelocityCommandCfg", env: Any):
        super().__init__(cfg, env)
        # Tensor buffer: [num_envs, 5] -> (vx, vy, wz, roll_cmd, pitch_cmd)
        self.command = torch.zeros((self.num_envs, 5), device=self.device)

    @property
    def command_dim(self) -> int:
        return 5

    def _resample_command(self, env_ids: torch.Tensor):
        n = len(env_ids)
        # Velocity targets
        self.command[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*self.cfg.lin_vel_x)
        self.command[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*self.cfg.lin_vel_y)
        self.command[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*self.cfg.ang_vel_z)

        # Attitude lean targets (radians)
        # 50% probability of neutral posture (0 rad), 50% sampled lean
        sample_mask = torch.rand(n, device=self.device) > 0.5
        rolls = torch.zeros(n, device=self.device)
        pitches = torch.zeros(n, device=self.device)

        rolls[sample_mask] = torch.empty(sample_mask.sum(), device=self.device).uniform_(*self.cfg.roll_range)
        pitches[sample_mask] = torch.empty(sample_mask.sum(), device=self.device).uniform_(*self.cfg.pitch_range)

        self.command[env_ids, 3] = rolls
        self.command[env_ids, 4] = pitches


@configclass
class AttitudeVelocityCommandCfg(CommandTermCfg):
    class_type: type = AttitudeVelocityCommand
    resampling_time_range: tuple[float, float] = (4.0, 6.0)
    lin_vel_x: tuple[float, float] = (-0.2, 0.8)
    lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    roll_range: tuple[float, float] = (-math.radians(12.0), math.radians(12.0))
    pitch_range: tuple[float, float] = (-math.radians(12.0), math.radians(12.0))


# ==============================================================================
# 2. Reward Functions for Posture Decoupling
# ==============================================================================

def track_commanded_attitude_exp(env: Any, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalizes squared error between commanded and actual roll/pitch derived from projected gravity."""
    asset = env.scene[asset_cfg.name]
    cmd: torch.Tensor = env.command_manager.get_command(command_name)  # [N, 5]

    # Extract projected gravity in base frame (gx, gy, gz)
    proj_g = asset.data.projected_gravity_b
    current_pitch = torch.atan2(-proj_g[:, 0], -proj_g[:, 2])
    current_roll = torch.atan2(proj_g[:, 1], -proj_g[:, 2])

    roll_err = torch.square(cmd[:, 3] - current_roll)
    pitch_err = torch.square(cmd[:, 4] - current_pitch)
    return torch.exp(-(roll_err + pitch_err) / (std ** 2))


# ==============================================================================
# 3. Manager Configurations
# ==============================================================================

@configclass
class CommandsCfg:
    base_velocity_attitude = AttitudeVelocityCommandCfg()


@configclass
class EventCfg:
    """Domain randomizations: Physics properties and disturbance wrenches."""

    # Randomize trunk center of mass on episode reset
    randomize_base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.03, 0.03),  # +/- 30 mm
                "y": (-0.015, 0.015),
                "z": (-0.02, 0.02),
            },
        },
    )

    # Periodic disturbance forces/torques applied directly to the head/neck mount
    neck_disturbance_wrench = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.3, 0.8),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="neck_link"),
            "force_range": {
                "x": (-3.0, 3.0),
                "y": (-3.0, 3.0),
                "z": (-1.5, 1.5),
            },
            "torque_range": {
                "x": (-0.8, 0.8),
                "y": (-0.8, 0.8),
                "z": (-0.5, 0.5),
            },
        },
    )

    # Base payload mass variation
    randomize_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.5, 0.5),  # +/- 0.5 kg
            "operation": "add",
        },
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        last_action = ObservationTermCfg(func=mdp.last_action)

        commands = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity_attitude"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # 1. Primary task: Track linear and angular velocity
    track_lin_vel = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"std": 0.25, "command_name": "base_velocity_attitude"},
    )
    track_ang_vel = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"std": 0.25, "command_name": "base_velocity_attitude"},
    )

    # 2. Secondary task: Track commanded roll/pitch offsets
    track_posture = RewardTermCfg(
        func=track_commanded_attitude_exp,
        weight=0.6,
        params={
            "std": 0.35,
            "command_name": "base_velocity_attitude",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 3. Regularization: Penalize excessive torque and acceleration
    joint_torques = RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.0001)
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)


# ==============================================================================
# 4. Root Environment Configuration
# ==============================================================================

@configclass
class MicroduckLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneEntityCfg = SceneEntityCfg()
    robot: ArticulationCfg = MISSING

    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        self.decimation = 4  # 200 Hz sim / 4 = 50 Hz control loop
        self.episode_length_s = 20.0
