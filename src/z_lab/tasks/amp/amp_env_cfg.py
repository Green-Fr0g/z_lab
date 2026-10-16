"""Public AMP locomotion environment configuration.

Robot-specific assets, names, action scales, and motion directories are intentionally
left for configurations under ``tasks/amp/config/<robot>``.
"""

from __future__ import annotations

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
    """Flat scene shared by all AMP robot configurations."""

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
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
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
class CommandsCfg:
    """Velocity command used by the generic AMP locomotion task."""

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
            ang_vel_z=(-1.5, 1.5),
            heading=(-math.pi / 2, math.pi / 2),
        ),
    )


@configclass
class ActionsCfg:
    """Generic joint-position action; robot configs provide the actual order and scale."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Policy, privileged critic, and discriminator observations."""

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
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 4

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))
        body_pos_b = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
        )
        body_ori_b = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 4

    @configclass
    class AmpPolicyCfg(ObsGroup):
        """Current simulated state supplied to an AMP discriminator."""

        body_pos_b = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
        )
        body_ori_b = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
        )
        body_lin_vel_b = ObsTerm(
            func=mdp.robot_body_lin_vel_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
        )
        body_ang_vel_b = ObsTerm(
            func=mdp.robot_body_ang_vel_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 4

    @configclass
    class AmpExpertCfg(ObsGroup):
        """Online reference state supplied to an AMP discriminator."""

        body_pos_b = ObsTerm(
            func=mdp.motion_body_pos_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
        )
        body_ori_b = ObsTerm(
            func=mdp.motion_body_ori_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
        )
        body_lin_vel_b = ObsTerm(
            func=mdp.motion_body_lin_vel_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
        )
        body_ang_vel_b = ObsTerm(
            func=mdp.motion_body_ang_vel_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),
            },
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 4

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp_policy: AmpPolicyCfg = AmpPolicyCfg()
    amp_expert: AmpExpertCfg = AmpExpertCfg()


@configclass
class EventsCfg:
    """Motion reset and generic AMP domain-randomization events."""

    init_motion_loader = EventTerm(
        func=mdp.init_motion_loader,
        mode="startup",
        params={
            "motion_dir": "",
            "root_name": "",
            "all_body_names": (),
            "recovery_motion_dir": None,
            "recovery_name_patterns": ("fallAndGetUp",),
        },
    )
    reset_from_motion = EventTerm(
        func=mdp.reset_from_motion_data,
        mode="reset",
        params={
            "motion_dir": "",
            "root_name": "",
            "all_body_names": (),
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
            "recovery_motion_dir": None,
            "recovery_env_ratio": 0.4,
            "recovery_name_patterns": ("fallAndGetUp",),
            "recovery_max_steps": 250,
        },
    )
    advance_motion_reference = EventTerm(
        func=mdp.advance_motion_reference,
        mode="interval",
        interval_range_s=(0.02, 0.02),
    )
    update_recovery_status = EventTerm(
        func=mdp.update_recovery_status,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        # Robot configurations must provide the appropriate standing height.
        params={"success_height": 0.0, "success_angle": 0.61, "success_hold_steps": 10},
    )
    recovery_assist_force = EventTerm(
        func=mdp.RecoveryAssistForceTerm,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=()),
            "force_start": 0.0,
            "force_end": 0.0,
            "anneal_steps": 50000,
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
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
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
            }
        },
    )


@configclass
class RewardsCfg:
    """Generic locomotion rewards; robot configurations may override weights."""

    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)
    track_anchor_linear_velocity = RewTerm(
        func=mdp.track_anchor_linear_velocity,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 1.0, "anchor_cfg": SceneEntityCfg("robot", body_names="")},
    )
    track_anchor_angular_velocity = RewTerm(
        func=mdp.track_anchor_angular_velocity,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.pi, "anchor_cfg": SceneEntityCfg("robot", body_names="")},
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=""), "target_height": 0.78},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )
    body_ang_vel_xy_l2 = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=0.0,
        params={"body_cfg": SceneEntityCfg("robot", body_names="")},
    )
    recovery_success_rate = RewTerm(func=mdp.recovery_success_rate, weight=0.0)


@configclass
class TerminationsCfg:
    """Generic AMP episode termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.recovery_aware_illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )
    base_height = DoneTerm(func=mdp.recovery_aware_root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.recovery_aware_bad_orientation, params={"limit_angle": 0.7})
    recovery_failure = DoneTerm(func=mdp.recovery_failure)


@configclass
class CurriculumCfg:
    """Curriculum terms reserved for robot-specific configurations."""

    pass


@configclass
class AmpFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Public flat AMP environment with robot-specific values left injectable."""

    scene: AmpSceneCfg = AmpSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = mdp.AmpPhysicsCfg()
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

    def play_mode(self):
        """Disable stochastic disturbances while preserving motion-based reset."""
        super().play_mode()
        self.events.randomize_rigid_body_material = None
        self.events.randomize_com_positions = None
        self.events.randomize_actuator_gains = None
        self.events.push_robot = None
