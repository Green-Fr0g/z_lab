"""AMP configuration for the Unitree G1."""

from pathlib import Path

from isaaclab.utils import configclass

from z_lab.assets.robots.g1 import (
    G1_ACTION_SCALE,
    G1_AMP_DOF_ORDER,
    G1_AMP_LINK_NAMES,
    G1_ANCHOR_LINK_NAME,
    G1_BODY_ORDER,
    G1_CFG,
    G1_ROOT_LINK_NAME,
)
from z_lab.tasks.amp.amp_env_cfg import AmpEnvCfg

G1_AMP_MOTION_DIR = str(Path(__file__).resolve().parent / "motions")
"""Recorded clips in the ``csv_to_npz.py`` schema."""

G1_AMP_ALL_BODY_NAMES = tuple(G1_BODY_ORDER)
"""Every body the recorded clips contain, independent of the feature subset."""


@configclass
class UnitreeG1AmpEnvCfg(AmpEnvCfg):
    """G1 velocity-conditioned AMP on flat terrain."""

    base_link_name = "torso_link"
    root_link_name = G1_ROOT_LINK_NAME
    foot_link_name = ".*_ankle_roll_link"
    anchor_body_name = G1_ANCHOR_LINK_NAME
    motion_dir = G1_AMP_MOTION_DIR
    amp_all_body_names = G1_AMP_ALL_BODY_NAMES
    joint_names = tuple(G1_AMP_DOF_ORDER)
    amp_body_names = tuple(G1_AMP_LINK_NAMES)

    def __post_init__(self) -> None:
        super().__post_init__()

        # ------------------------------ scene ------------------------------
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------ actions ------------------------------
        self.actions.joint_pos.joint_names = list(self.joint_names)
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        # ------------------------------ commands ------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.57, 1.57)
        self.commands.base_velocity.ranges.heading = (-1.57, 1.57)

        # ------------------------------ events ------------------------------
        self.events.init_motion_loader.params["motion_dir"] = self.motion_dir
        self.events.init_motion_loader.params["root_name"] = self.root_link_name
        self.events.init_motion_loader.params["all_body_names"] = self.amp_all_body_names
        self.events.reset_from_motion.params["motion_dir"] = self.motion_dir
        self.events.reset_from_motion.params["root_name"] = self.root_link_name
        self.events.reset_from_motion.params["all_body_names"] = self.amp_all_body_names
        self.events.reset_from_motion.params["asset_cfg"].joint_names = list(self.joint_names)

        # ------------------------------ observations ------------------------------
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = list(self.joint_names)
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = list(self.joint_names)
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = list(self.joint_names)
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = list(self.joint_names)

        self.observations.critic.body_pos_b.params["anchor_cfg"].body_names = (self.anchor_body_name,)
        self.observations.critic.body_pos_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.critic.body_ori_b.params["anchor_cfg"].body_names = (self.anchor_body_name,)
        self.observations.critic.body_ori_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp_state.body_pos_b.params["anchor_cfg"].body_names = (self.anchor_body_name,)
        self.observations.amp_state.body_pos_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp_state.body_ori_b.params["anchor_cfg"].body_names = (self.anchor_body_name,)
        self.observations.amp_state.body_ori_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp_state.body_lin_vel_b.params["anchor_cfg"].body_names = (self.anchor_body_name,)
        self.observations.amp_state.body_lin_vel_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp_state.body_ang_vel_b.params["anchor_cfg"].body_names = (self.anchor_body_name,)
        self.observations.amp_state.body_ang_vel_b.params["body_cfg"].body_names = list(self.amp_body_names)

        # ------------------------------ rewards ------------------------------
        self.rewards.is_terminated.weight = -200.0
        self.rewards.track_anchor_linear_velocity.weight = 1.0
        self.rewards.track_anchor_linear_velocity.params["anchor_cfg"].body_names = self.anchor_body_name
        self.rewards.track_anchor_angular_velocity.weight = 1.0
        self.rewards.track_anchor_angular_velocity.params["anchor_cfg"].body_names = self.anchor_body_name
        self.rewards.body_ang_vel_xy_l2.weight = -0.5
        self.rewards.body_ang_vel_xy_l2.params["body_cfg"].body_names = "pelvis"
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.base_height_l2.weight = -5.0
        self.rewards.joint_pos_limits.weight = -0.5
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.feet_slide.weight = -0.2
        self.rewards.undesired_contacts.weight = -1.0

        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_height_l2.params["target_height"] = 0.78
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = list(self.joint_names)
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["(?!.*ankle_roll).*"]

        # ------------------------------ terminations ------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.base_height.params["minimum_height"] = 0.2
        self.terminations.bad_orientation.params["limit_angle"] = 0.7

        self.disable_zero_weight_rewards()
