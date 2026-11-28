"""Reusable manager terms, state observations, and physics presets for SuperAMP."""

from isaaclab.envs.mdp import (
    action_rate_l2,
    bad_orientation,
    base_ang_vel,
    generated_commands,
    joint_acc_l2,
    joint_pos_limits,
    joint_vel_rel,
    last_action,
    projected_gravity,
    push_by_setting_velocity,
    time_out,
)

from z_lab.tasks.amp.mdp.rewards import body_ang_vel_xy_l2, track_anchor_angular_velocity
from z_lab.tasks.supermimic.mdp.actions import BiasedJointPositionActionCfg
from z_lab.tasks.supermimic.mdp.events import randomize_encoder_bias, randomize_foot_friction, randomize_torso_com
from z_lab.tasks.supermimic.mdp.observations import biased_joint_pos_rel
from z_lab.tasks.supermimic.mdp.physics import SimplePhysicsCfg

from .commands import SuperAmpVelocityCommandCfg
from .curriculums import velocity_levels
from .events import ResetFromExpertMotion
from .observations import gencoord_state, maxcoord_state
from .rewards import track_anchor_lin_vel_xy
