"""Shared manager-based environments for SuperMimic motion tracking."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from z_lab.tasks import mimic

from . import mdp


@configclass
class CommandsCfg:
    """Reference motions with episode reset perturbations."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range={
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        },
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    """Joint position control in policy joint order."""

    joint_pos = mdp.BiasedJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], preserve_order=True, use_default_offset=True
    )


@configclass
class EventsCfg:
    """Per-environment domain randomization at simulator startup."""

    foot_friction = EventTerm(
        func=mdp.randomize_foot_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_range": (0.3, 1.2),
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_torso_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    encoder_bias = EventTerm(
        func=mdp.randomize_encoder_bias,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    )


@configclass
class ActorCfg(ObsGroup):
    """Proprioception in policy joint order."""

    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    joint_pos = ObsTerm(
        func=mdp.biased_joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)},
    )
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GoalCfg(ObsGroup):
    """Current and future reference joint states and relative root poses."""

    command = ObsTerm(
        func=mdp.motion_joint_window,
        params={"num_frames": 10, "frame_stride": 5},
    )
    motion_anchor_pos_b = ObsTerm(
        func=mdp.motion_anchor_pos_window,
        params={"num_frames": 10, "frame_stride": 5},
    )
    motion_anchor_ori_b = ObsTerm(
        func=mdp.motion_anchor_ori_window,
        params={"num_frames": 10, "frame_stride": 5},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GenCoordStateCfg(ObsGroup):
    """Floating-base state and articulated joint state."""

    state = ObsTerm(
        func=mdp.gencoord_state,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class MaxCoordStateCfg(ObsGroup):
    """Tracked link poses and link velocities."""

    state = ObsTerm(func=mdp.maxcoord_state)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    """Observation groups consumed by SuperMimic."""

    actor: ActorCfg = ActorCfg()
    full_state: GenCoordStateCfg | MaxCoordStateCfg = GenCoordStateCfg()
    goal: GoalCfg = GoalCfg()


@configclass
class RewardsCfg:
    """Tracking and regularization terms shared by both representations."""

    motion_global_anchor_pos = RewTerm(
        func=mimic.mdp.motion_global_anchor_position_error_exp,
        weight=0.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mimic.mdp.motion_global_anchor_orientation_error_exp,
        weight=0.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mimic.mdp.motion_relative_body_position_error_exp,
        weight=0.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mimic.mdp.motion_relative_body_orientation_error_exp,
        weight=0.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mimic.mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mimic.mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mimic.mdp.action_rate_l2, weight=0.0)
    joint_limit = RewTerm(
        func=mimic.mdp.joint_pos_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class TerminationsCfg:
    """Root tracking failures, time limits, and reference clip boundaries."""

    time_out = DoneTerm(func=mimic.mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mimic.mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mimic.mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    motion_end = DoneTerm(func=mdp.end_of_motion, time_out=True)


@configclass
class SuperMimicGenCoordEnvCfg(ManagerBasedRLEnvCfg):
    """Shared SuperMimic environment with generalized-coordinate observations."""

    scene: mimic.MimicSceneCfg = mimic.MimicSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: mimic.CurriculumCfg = mimic.CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = mdp.SimplePhysicsCfg()
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

    def play_mode(self) -> None:
        """Use one environment and fixed startup parameters for playback."""
        super().play_mode()
        self.scene.num_envs = 1
        self.commands.motion.sampling_mode = "start"
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.joint_position_range = (0.0, 0.0)
        self.events.foot_friction.params["friction_range"] = (0.75, 0.75)
        self.events.base_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        self.events.encoder_bias.params["bias_range"] = (0.0, 0.0)


@configclass
class SuperMimicMaxCoordEnvCfg(SuperMimicGenCoordEnvCfg):
    """Shared SuperMimic environment with tracked link observations."""

    observations: ObservationsCfg = ObservationsCfg(full_state=MaxCoordStateCfg())
