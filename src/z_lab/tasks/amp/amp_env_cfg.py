# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared AMP configuration for the Unitree robots.

The environment owns locomotion rewards, motion-based resets, and a single raw ``amp_state``
observation group. Expert sampling and the discriminator belong to the training layer, so nothing
here depends on the AMP algorithm implementation.
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
    """Flat scene for AMP locomotion."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
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
    """Command specifications for velocity-conditioned AMP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.25,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 3.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-math.pi / 2, math.pi / 2),
            heading=(-math.pi / 2, math.pi / 2),
        ),
    )


@configclass
class AmpActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class AmpObservationsCfg:
    """Observation specifications for AMP flat locomotion."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 4

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        body_pos_b = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        body_ori_b = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 4

    @configclass
    class AmpStateCfg(ObsGroup):
        body_pos_b = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        body_ori_b = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        body_lin_vel_b = ObsTerm(
            func=mdp.robot_body_lin_vel_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        body_ang_vel_b = ObsTerm(
            func=mdp.robot_body_ang_vel_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp_state: AmpStateCfg = AmpStateCfg()


@configclass
class AmpEventsCfg:
    """Motion loading, motion resets, and the usual randomizations."""

    init_motion_loader = EventTerm(
        func=mdp.init_motion_loader,
        mode="startup",
        params={"motion_dir": "", "root_name": "", "all_body_names": ()},  # set per robot
    )
    reset_from_motion = EventTerm(
        func=mdp.reset_from_motion_data,
        mode="reset",
        params={
            "motion_dir": "",  # set per robot
            "root_name": "",  # set per robot
            "all_body_names": (),  # set per robot
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
        },
    )
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.0),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )


@configclass
class AmpRewardsCfg:
    """Reward terms for the MDP."""

    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_anchor_linear_velocity = RewTerm(
        func=mdp.track_anchor_linear_velocity,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 1.0, "anchor_cfg": SceneEntityCfg("robot", body_names="")},
    )
    track_anchor_angular_velocity = RewTerm(
        func=mdp.track_anchor_angular_velocity,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 3.14, "anchor_cfg": SceneEntityCfg("robot", body_names="")},
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "sensor_cfg": None,
            "target_height": 0.78,
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )
    body_ang_vel_xy_l2 = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=0.0,
        params={"std": 3.14, "body_cfg": SceneEntityCfg("robot", body_names="")},
    )


@configclass
class AmpTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})


@configclass
class AmpEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the velocity-conditioned AMP flat environment."""

    scene: AmpSceneCfg = AmpSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: AmpObservationsCfg = AmpObservationsCfg()
    actions: AmpActionsCfg = AmpActionsCfg()
    commands: AmpCommandsCfg = AmpCommandsCfg()
    rewards: AmpRewardsCfg = AmpRewardsCfg()
    terminations: AmpTerminationsCfg = AmpTerminationsCfg()
    events: AmpEventsCfg = AmpEventsCfg()

    def __post_init__(self) -> None:
        self.compute_final_obs = True
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = mdp.AmpPhysicsCfg()
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt

    def play_mode(self) -> None:
        """Disable the interval push event so evaluation is not perturbed."""
        super().play_mode()
        self.events.push_robot = None