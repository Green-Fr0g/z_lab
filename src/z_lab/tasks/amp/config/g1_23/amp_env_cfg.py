"""G1_23 environment settings for the AMP_mjlab flat locomotion task."""

from pathlib import Path

from isaaclab.utils import configclass

from z_lab.assets.robots.g1_23 import (
    G1_23DOF_ACTION_SCALE,
    G1_23DOF_AMP_DOF_ORDER,
    G1_23DOF_AMP_LINK_NAMES,
    G1_23DOF_ANCHOR_LINK_NAME,
    G1_23DOF_CFG,
    G1_23DOF_ROOT_LINK_NAME,
)
from z_lab.tasks.amp.amp_env_cfg import AmpEnvCfg

MOTION_DIR = Path(__file__).resolve().parent / "motions"
RECOVERY_CLIP_NAME = "fallAndGetUp1_subject1_frames409_1948_g1_23"


@configclass
class UnitreeG123AmpEnvCfg(AmpEnvCfg):
    """Flat velocity-conditioned AMP using the prepared 23-DOF G1 motion set."""

    amp_body_names = G1_23DOF_AMP_LINK_NAMES
    joint_names = G1_23DOF_AMP_DOF_ORDER
    anchor_link_name = G1_23DOF_ANCHOR_LINK_NAME
    root_link_name = G1_23DOF_ROOT_LINK_NAME
    foot_links = "left_ankle_roll_link", "right_ankle_roll_link"
    amp_action_scale = G1_23DOF_ACTION_SCALE

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------scene------------------------------
        self.scene.robot = G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------actions------------------------------
        self.actions.joint_pos.joint_names = list(self.joint_names)
        self.actions.joint_pos.scale = self.amp_action_scale

        # ------------------------------observations------------------------------

        # actor related
        self.observations.actor.joint_pos.params["asset_cfg"].joint_names = list(self.joint_names)
        self.observations.actor.joint_vel.params["asset_cfg"].joint_names = list(self.joint_names)

        # critic related
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = list(self.joint_names)
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = list(self.joint_names)
        self.observations.critic.body_pos_b.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.observations.critic.body_pos_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.critic.body_ori_b.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.observations.critic.body_ori_b.params["body_cfg"].body_names = list(self.amp_body_names)

        # AMP discriminator related
        self.observations.amp.body_pos_b.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.observations.amp.body_pos_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp.body_ori_b.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.observations.amp.body_ori_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp.body_lin_vel_b.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.observations.amp.body_lin_vel_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp.body_ang_vel_b.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.observations.amp.body_ang_vel_b.params["body_cfg"].body_names = list(self.amp_body_names)

        # ------------------------------rewards------------------------------

        # termination related
        self.rewards.is_terminated.weight = -200.0

        # track related
        self.rewards.track_anchor_linear_velocity.weight = 1.0
        self.rewards.track_anchor_angular_velocity.weight = 1.0
        self.rewards.track_anchor_linear_velocity.params["delay_env_rew_ratio"] = 0.0
        self.rewards.track_anchor_angular_velocity.params["delay_env_rew_ratio"] = 0.0

        # root related
        self.rewards.track_root_height.weight = 1.0
        self.rewards.body_ang_vel_xy_l2.weight = 0.5
        self.rewards.track_root_height.params["delay_env_rew_ratio"] = 3.5
        self.rewards.body_ang_vel_xy_l2.params["delay_env_rew_ratio"] = 0.0

        # joint related
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.action_rate_l2.weight = -0.01

        # feet related
        self.rewards.feet_slide.weight = -0.25
        self.rewards.undesired_contacts.weight = -0.1

        # ------------------------------params------------------------------

        # track related
        self.rewards.track_anchor_linear_velocity.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.rewards.track_anchor_angular_velocity.params["anchor_cfg"].body_names = [self.anchor_link_name]

        # root related
        self.rewards.track_root_height.params["asset_cfg"].body_names = [self.root_link_name]
        self.rewards.body_ang_vel_xy_l2.params["body_cfg"].body_names = [self.root_link_name]

        # joint related
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = list(self.joint_names)

        # feet related
        self.rewards.feet_slide.params["asset_cfg"].body_names = list(self.foot_links)
        self.rewards.feet_slide.params["sensor_cfg"].body_names = list(self.foot_links)
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["(?!.*ankle).*"]

        # ------------------------------events------------------------------

        # reference motion initialization and reset
        self.events.init_motion_loader.params["motion_dir"] = str(MOTION_DIR)
        self.events.init_motion_loader.params["recovery_motion_names"] = (RECOVERY_CLIP_NAME,)
        self.events.reset_from_motion.params["motion_dir"] = str(MOTION_DIR)
        self.events.reset_from_motion.params["recovery_motion_names"] = (RECOVERY_CLIP_NAME,)
        self.events.reset_from_motion.params["root_name"] = self.root_link_name

        # recovery assistance: world +Z force in N, decayed over control steps
        self.events.recovery_assist.params["asset_cfg"].body_names = [self.root_link_name]
        self.events.recovery_assist.params["initial_force"] = 200.0
        self.events.recovery_assist.params["decay_steps"] = 120_000

        # feet material randomization
        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = list(self.foot_links)
