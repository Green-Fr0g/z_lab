"""Manager-based AMP locomotion environment, independent of the training algorithm."""

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
from z_lab.tasks.locomotion.mdp.physics import SimplePhysicsCfg


@configclass
class AmpSceneCfg(InteractiveSceneCfg):
    """Flat Newton scene for AMP locomotion."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.38, 0.4)),
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAACLAB_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class AmpCommandsCfg:
    """Initial velocity ranges and final limits for the linear curriculum."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 3.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.57, 1.57),
        ),
    )


@configclass
class AmpActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class AmpObservationsCfg:
    @configclass
    class ActorCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(
            func=mdp.last_action,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 4

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        actions = ObsTerm(
            func=mdp.last_action,
        )
        body_pos_b = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", preserve_order=True),
            },
        )
        body_ori_b = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", preserve_order=True),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 4

    @configclass
    class AmpCfg(ObsGroup):
        body_pos_b = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", preserve_order=True),
            },
        )
        body_ori_b = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", preserve_order=True),
            },
        )
        body_lin_vel_b = ObsTerm(
            func=mdp.robot_body_lin_vel_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", preserve_order=True),
            },
        )
        body_ang_vel_b = ObsTerm(
            func=mdp.robot_body_ang_vel_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", preserve_order=True),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    actor: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()
    amp: AmpCfg = AmpCfg()


@configclass
class AmpEventsCfg:
    init_motion_loader = EventTerm(
        func=mdp.init_motion_loader,
        mode="startup",
        params={
            "motion_dir": "",
            "recovery_motion_names": (),
            "delay_reset_env_ratio": 0.4,
            "max_delay_steps": 250,
        },
    )
    reset_from_motion = EventTerm(
        func=mdp.reset_from_motion_data,
        mode="reset",
        params={
            "motion_dir": "",
            "root_name": "",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
            "recovery_motion_names": (),
        },
    )
    recovery_assist = EventTerm(
        func=mdp.RecoveryAssistForce,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
        params={
            "initial_force": 0.0,
            "decay_steps": 120_000,
            "asset_cfg": SceneEntityCfg("robot", body_names=()),
        },
    )
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            }
        },
    )


@configclass
class AmpCurriculumCfg:
    recovery_assist_force = CurrTerm(
        func=mdp.recovery_assist_force,
    )
    command_vel = CurrTerm(
        func=mdp.commands_vel,
        params={
            "command_name": "base_velocity",
            "num_steps": 5000 * 24,
        },
    )


@configclass
class AmpRewardsCfg:
    """Shared reward terms; robot-specific configurations set their active weights."""

    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=0.0,
    )
    track_anchor_linear_velocity = RewTerm(
        func=mdp.track_anchor_linear_velocity,
        weight=0.0,
        params={
            "std": 1.0,
            "command_name": "base_velocity",
            "anchor_cfg": SceneEntityCfg("robot", body_names=()),
            "mask_delay": True,
        },
    )
    track_anchor_angular_velocity = RewTerm(
        func=mdp.track_anchor_angular_velocity,
        weight=0.0,
        params={
            "std": 3.14,
            "command_name": "base_velocity",
            "anchor_cfg": SceneEntityCfg("robot", body_names=()),
            "mask_delay": True,
        },
    )
    track_root_height = RewTerm(
        func=mdp.track_root_height,
        weight=0.0,
        params={
            "std": 0.3,
            "asset_cfg": SceneEntityCfg("robot", body_names=()),
            "mask_delay": True,
        },
    )
    body_ang_vel_xy_l2 = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=0.0,
        params={
            "std": 3.14,
            "body_cfg": SceneEntityCfg("robot", body_names=()),
            "mask_delay": True,
        },
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0,
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=0.0,
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=()),
            "asset_cfg": SceneEntityCfg("robot", body_names=()),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 10.0},
    )


@configclass
class AmpTerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )
    bad_base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5},
    )


@configclass
class AmpEnvCfg(ManagerBasedRLEnvCfg):
    scene: AmpSceneCfg = AmpSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )
    observations: AmpObservationsCfg = AmpObservationsCfg()
    actions: AmpActionsCfg = AmpActionsCfg()
    commands: AmpCommandsCfg = AmpCommandsCfg()
    events: AmpEventsCfg = AmpEventsCfg()
    rewards: AmpRewardsCfg = AmpRewardsCfg()
    terminations: AmpTerminationsCfg = AmpTerminationsCfg()
    curriculum: AmpCurriculumCfg = AmpCurriculumCfg()

    def __post_init__(self):
        self.compute_final_obs = True
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = SimplePhysicsCfg()
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
