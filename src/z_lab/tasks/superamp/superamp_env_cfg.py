"""Shared manager-based environments for SuperAMP locomotion."""

import math
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from z_lab.tasks.amp.amp_env_cfg import AmpSceneCfg
from z_lab.tasks.supermimic.tracking_env_cfg import ActionsCfg
from z_lab.tasks.supermimic.tracking_env_cfg import EventsCfg as SuperMimicEventsCfg

from . import mdp


@configclass
class SceneCfg(AmpSceneCfg):
    """Flat locomotion scene with local lighting."""


@configclass
class CommandsCfg:
    """Velocity command sampling and ranges for the curriculum."""

    base_velocity = mdp.SuperAmpVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        rel_lateral_envs=0.10,
        rel_turning_envs=0.10,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.SuperAmpVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.4),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.5, 0.5),
        ),
        limit_ranges=mdp.SuperAmpVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 3.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.57, 1.57),
        ),
    )


@configclass
class EventsCfg(SuperMimicEventsCfg):
    """Randomize dynamics at startup and initialize each episode from an expert frame."""

    reset_from_expert = EventTerm(
        func=mdp.ResetFromExpertMotion,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "motion_file": MISSING,
            "root_name": MISSING,
            "exclude_motion_names": (),
        },
    )
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 3.0),
    #     params={
    #         "velocity_range": {
    #             "x": (-1.0, 1.0),
    #             "y": (-0.5, 0.5),
    #             "z": (-0.4, 0.4),
    #             "roll": (-0.52, 0.52),
    #             "pitch": (-0.52, 0.52),
    #             "yaw": (-0.78, 0.78),
    #         }
    #     },
    # )


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
    """Commanded planar velocity and yaw rate."""

    velocity = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GenCoordStateCfg(ObsGroup):
    """Floating-base and joint state in the configured joint order."""

    state = ObsTerm(
        func=mdp.gencoord_state,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class MaxCoordStateCfg(ObsGroup):
    """Tracked link poses and velocities in the robot-specific link order."""

    state = ObsTerm(
        func=mdp.maxcoord_state,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*"], preserve_order=True)},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    """Proprioception, world-model state, and velocity goal."""

    actor: ActorCfg = ActorCfg()
    full_state: GenCoordStateCfg | MaxCoordStateCfg = GenCoordStateCfg()
    goal: GoalCfg = GoalCfg()


@configclass
class RewardsCfg:
    """Velocity tracking and regularization terms available to the world model."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_anchor_lin_vel_xy,
        weight=0.0,
        params={"std": 1.0, "command_name": "base_velocity", "anchor_cfg": SceneEntityCfg("robot")},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_anchor_angular_velocity,
        weight=0.0,
        params={"std": 1.0, "command_name": "base_velocity", "anchor_cfg": SceneEntityCfg("robot")},
    )
    body_ang_vel_xy_l2 = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=0.0,
        params={"std": 1.0, "body_cfg": SceneEntityCfg("robot")},
    )
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    """End an episode at the time limit or when the root tilts too far."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})


@configclass
class CurriculumCfg:
    """Expand linear and angular velocity commands as their tracking improves."""

    velocity_levels = CurrTerm(func=mdp.velocity_levels)


@configclass
class SuperAmpGenCoordEnvCfg(ManagerBasedRLEnvCfg):
    """AMP locomotion with a generalized-coordinate world-model state.

    Robot configurations bind the asset, ordered joints and links, expert motions,
    and reward weights as in the corresponding AMP environment.
    """

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.compute_final_obs = True
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = mdp.SimplePhysicsCfg()
        self.sim.physics_material = self.scene.terrain.physics_material

    def play_mode(self) -> None:
        """Sample playback commands over the final curriculum ranges."""
        super().play_mode()
        self.scene.num_envs = 20
        self.curriculum.velocity_levels = None

        base_velocity = self.commands.base_velocity
        play_ranges = base_velocity.limit_ranges.copy()
        if base_velocity.heading_command and play_ranges.heading is None:
            play_ranges.heading = base_velocity.ranges.heading
        base_velocity.ranges = play_ranges

        self.events.foot_friction.params["friction_range"] = (0.75, 0.75)
        self.events.base_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        self.events.encoder_bias.params["bias_range"] = (0.0, 0.0)


@configclass
class SuperAmpMaxCoordEnvCfg(SuperAmpGenCoordEnvCfg):
    """AMP locomotion with a maximal-coordinate world-model state."""

    observations: ObservationsCfg = ObservationsCfg(full_state=MaxCoordStateCfg())
