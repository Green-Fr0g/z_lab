# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared AMP configuration for the Unitree robots.

The motion startup event loads raw reference states for resets. A separate startup event installs
delayed terminations for recovery training. Discriminator data preparation belongs to the training
layer, independently of these environment lifecycle components.
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from z_lab.tasks.amp import mdp


@configclass
class AmpSceneCfg(InteractiveSceneCfg):
    """Flat ground, one robot, and the contact sensor the foot rewards read."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned"
            "/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAACLAB_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class AmpCommandsCfg:
    """Velocity command with a fraction of turn-only environments."""

    base_velocity = mdp.AmpVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.02,
        rel_turn_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.AmpVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.1, 0.1),
            limit_lin_vel_x=(-1.0, 1.5),
            limit_lin_vel_y=(-1.0, 1.0),
            limit_ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class AmpActionsCfg:
    """Every joint of the robot, resolved from the wildcard expression."""

    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class AmpObservationsCfg:
    """Policy, critic, and the discriminator input group."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AmpPolicyCfg(ObsGroup):
        """Single raw frame of all bodies/joints; AmpDataset owns the shared AMP features."""

        body_pos_w = ObsTerm(func=mdp.robot_body_pos_w)
        body_quat_w = ObsTerm(func=mdp.robot_body_quat_w)
        body_lin_vel_w = ObsTerm(func=mdp.robot_body_lin_vel_w)
        body_ang_vel_w = ObsTerm(func=mdp.robot_body_ang_vel_w)
        dof_pos = ObsTerm(func=mdp.robot_joint_pos)
        dof_vel = ObsTerm(func=mdp.robot_joint_vel)

        def __post_init__(self):
            self.history_length = 0
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp_policy: AmpPolicyCfg = AmpPolicyCfg()


@configclass
class AmpEventsCfg:
    """Motion loading, motion resets, delayed termination, and the usual randomization."""

    init_amp_motion = EventTerm(
        func=mdp.init_amp_motion,
        mode="startup",
        params={
            "motion_dir": "",  # set per robot
            "root_name": None,  # set per robot; independent of the AMP feature anchor
            "recovery_dir": None,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    init_delayed_termination = EventTerm(
        func=mdp.init_delayed_termination,
        mode="startup",
        params={"delay_reset_env_ratio": 0.0, "max_delay_steps": 0},
    )
    reset_from_motion = EventTerm(
        func=mdp.reset_from_amp_motion,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 6.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class AmpRewardsCfg:
    """Command tracking measured at the anchor body, plus recovery rewards and regularization."""

    track_anchor_linear_velocity = RewTerm(
        func=mdp.track_anchor_linear_velocity,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 1.0,
            "anchor_cfg": SceneEntityCfg("robot", body_names=()),  # set per robot
            "scale_in_recovery": True,
            "recovery_reward_scale": 0.0,
        },
    )
    track_anchor_angular_velocity = RewTerm(
        func=mdp.track_anchor_angular_velocity,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "std": 3.14,
            "anchor_cfg": SceneEntityCfg("robot", body_names=()),  # set per robot
            "scale_in_recovery": True,
            "recovery_reward_scale": 0.0,
        },
    )
    track_root_height = RewTerm(
        func=mdp.track_root_height,
        weight=1.0,
        params={
            "std": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
            "scale_in_recovery": True,
            # getting up is the whole point of the recovery window, so it pays more than walking
            "recovery_reward_scale": 3.5,
        },
    )
    body_ang_vel_xy_l2 = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=0.5,
        params={
            "std": 3.14,
            "body_cfg": SceneEntityCfg("robot", body_names=()),  # set per robot
            "scale_in_recovery": True,
            "recovery_reward_scale": 0.0,
        },
    )
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class AmpTerminationsCfg:
    """Timeouts plus the fall conditions the recovery logic watches."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)})


@configclass
class AmpCurriculumCfg:
    """Widen the command range once tracking saturates."""

    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)


@configclass
class AmpEnvCfg(ManagerBasedRLEnvCfg):
    """Shared AMP environment; concrete robots fill in the robot, motion, and body names."""

    scene: AmpSceneCfg = AmpSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: AmpObservationsCfg = AmpObservationsCfg()
    actions: AmpActionsCfg = AmpActionsCfg()
    commands: AmpCommandsCfg = AmpCommandsCfg()
    rewards: AmpRewardsCfg = AmpRewardsCfg()
    terminations: AmpTerminationsCfg = AmpTerminationsCfg()
    events: AmpEventsCfg = AmpEventsCfg()
    curriculum: AmpCurriculumCfg = AmpCurriculumCfg()

    amp_body_names: tuple[str, ...] = ()
    """Links that take part in the discriminator features."""

    amp_anchor_name: str = ""
    amp_joint_names: tuple[str, ...] = ()

    def __post_init__(self):
        self.compute_final_obs = True
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = mdp.AmpPhysicsCfg()
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt

    def play_mode(self):
        """Evaluate the widest command range without the training-time perturbations."""
        super().play_mode()
        self.events.push_robot = None
        # keep spawning from motion clips so the evaluated policy starts in-distribution
        self.events.reset_from_motion.params["use_recovery"] = False
        command_ranges = self.commands.base_velocity.ranges
        command_ranges.lin_vel_x = self.commands.base_velocity.ranges.limit_lin_vel_x
        command_ranges.lin_vel_y = self.commands.base_velocity.ranges.limit_lin_vel_y
        command_ranges.ang_vel_z = self.commands.base_velocity.ranges.limit_ang_vel_z
