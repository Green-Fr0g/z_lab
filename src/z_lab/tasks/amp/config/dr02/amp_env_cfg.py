"""Flat velocity-conditioned AMP locomotion for DR02."""

from pathlib import Path

from isaaclab.utils import configclass

from z_lab.assets.robots.dr02 import (
    DR02_ACTION_SCALE,
    DR02_AMP_DOF_ORDER,
    DR02_AMP_LINK_NAMES,
    DR02_ANCHOR_LINK_NAME,
    DR02_CFG,
    DR02_ROOT_LINK_NAME,
)
from z_lab.tasks.amp.amp_env_cfg import AmpEnvCfg

MOTION_DIR = Path(__file__).resolve().parent / "motions"
RECOVERY_CLIP_NAME = "fallAndGetUp1_subject1_frames409_1948"


@configclass
class DR02AmpEnvCfg(AmpEnvCfg):
    """Use the prepared DR02 reference motions for velocity tracking and recovery."""

    amp_body_names = DR02_AMP_LINK_NAMES
    joint_names = DR02_AMP_DOF_ORDER
    anchor_link_name = DR02_ANCHOR_LINK_NAME
    root_link_name = DR02_ROOT_LINK_NAME
    foot_links = ["left_ankle_x_link", "right_ankle_x_link"]
    amp_action_scale = DR02_ACTION_SCALE

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------scene------------------------------
        self.scene.robot = DR02_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------actions------------------------------
        self.actions.joint_pos.joint_names = self.joint_names
        self.actions.joint_pos.scale = self.amp_action_scale

        # ------------------------------observations------------------------------

        # actor related
        self.observations.actor.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.actor.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # critic related
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.body_pos_b.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.observations.critic.body_pos_b.params["body_cfg"].body_names = self.amp_body_names
        self.observations.critic.body_ori_b.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.observations.critic.body_ori_b.params["body_cfg"].body_names = self.amp_body_names

        # AMP discriminator related
        self.observations.amp.body_pos_b.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.observations.amp.body_pos_b.params["body_cfg"].body_names = self.amp_body_names
        self.observations.amp.body_ori_b.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.observations.amp.body_ori_b.params["body_cfg"].body_names = self.amp_body_names
        self.observations.amp.body_lin_vel_b.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.observations.amp.body_lin_vel_b.params["body_cfg"].body_names = self.amp_body_names
        self.observations.amp.body_ang_vel_b.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.observations.amp.body_ang_vel_b.params["body_cfg"].body_names = self.amp_body_names

        # ------------------------------rewards------------------------------

        self.rewards.is_terminated.weight = -200.0
        self.rewards.track_anchor_linear_velocity.weight = 1.0
        self.rewards.track_anchor_angular_velocity.weight = 1.0
        self.rewards.track_anchor_linear_velocity.params["delay_env_rew_ratio"] = 0.0
        self.rewards.track_anchor_angular_velocity.params["delay_env_rew_ratio"] = 0.0
        self.rewards.track_root_height.weight = 1.0
        self.rewards.track_body_upright.weight = -1.5
        self.rewards.body_ang_vel_xy_l2.weight = 0.5
        self.rewards.track_root_height.params["delay_env_rew_ratio"] = 3.5
        self.rewards.body_ang_vel_xy_l2.params["delay_env_rew_ratio"] = 0.0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_slide.weight = -0.25
        self.rewards.undesired_contacts.weight = -0.1

        # ------------------------------params------------------------------

        self.rewards.track_anchor_linear_velocity.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.rewards.track_anchor_angular_velocity.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.rewards.track_root_height.params["asset_cfg"].body_names = [self.root_link_name]
        self.rewards.track_body_upright.params["body_cfg"].body_names = [self.anchor_link_name]
        self.rewards.track_body_upright.params["height_cfg"].body_names = [self.root_link_name]
        self.rewards.body_ang_vel_xy_l2.params["body_cfg"].body_names = [self.root_link_name]
        self.rewards.track_root_height.params["std"] = 0.2
        self.rewards.track_body_upright.params["height_std"] = 0.2
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.feet_slide.params["asset_cfg"].body_names = self.foot_links
        self.rewards.feet_slide.params["sensor_cfg"].body_names = self.foot_links
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["(?!.*ankle_x_link).*"]

        # ------------------------------events------------------------------

        self.events.init_motion_loader.params["motion_dir"] = str(MOTION_DIR)
        self.events.init_motion_loader.params["recovery_motion_names"] = (RECOVERY_CLIP_NAME,)
        self.events.reset_from_motion.params["motion_dir"] = str(MOTION_DIR)
        self.events.reset_from_motion.params["recovery_motion_names"] = (RECOVERY_CLIP_NAME,)
        self.events.reset_from_motion.params["root_name"] = self.root_link_name
        self.events.recovery_assist.params["asset_cfg"].body_names = [self.anchor_link_name]
        self.events.recovery_assist.params["initial_force"] = 200.0
        self.events.recovery_assist.params["start_steps"] = 1000 * 24
        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = self.foot_links
