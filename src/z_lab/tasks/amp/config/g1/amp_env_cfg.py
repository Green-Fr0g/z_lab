"""Unitree G1 configuration for online-reference AMP experiments."""

from pathlib import Path

from isaaclab.utils import configclass

from z_lab.assets.robots.g1 import (
    G1_ACTION_SCALE,
    G1_AMP_LINK_NAMES,
    G1_ANCHOR_LINK_NAME,
    G1_CFG,
    G1_DOF_ORDER,
    G1_ROOT_LINK_NAME,
)
from z_lab.tasks.amp.amp_env_cfg import AmpFlatEnvCfg

MOTION_DIR = Path(__file__).resolve().parent / "motions"

G1_AMP_ALL_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)


@configclass
class G1AmpEnvCfg(AmpFlatEnvCfg):
    """G1 AMP environment with mixed walking and recovery resets."""

    motion_dir = str(MOTION_DIR)
    recovery_motion_dir = None
    root_link_name = G1_ROOT_LINK_NAME
    anchor_body_name = G1_ANCHOR_LINK_NAME
    amp_body_names = G1_AMP_LINK_NAMES
    amp_all_body_names = G1_AMP_ALL_BODY_NAMES
    joint_names = G1_DOF_ORDER
    recovery_env_ratio = 0.4
    recovery_name_patterns = ("fallAndGetUp",)
    recovery_max_steps = 250
    recovery_success_height = 0.65
    recovery_success_angle = 0.61
    recovery_success_hold_steps = 10
    assist_force_start = 200.0
    assist_force_end = 0.0
    # 20,000 PPO rollouts x 24 env steps per rollout.  The task reads
    # Isaac Lab's common_step_counter, so this remains independent of rsl_rl.
    assist_anneal_steps = 480_000

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.joint_names = list(G1_DOF_ORDER)
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        # Robot-specific motion-loader and observation name mappings.
        self.events.init_motion_loader.params["motion_dir"] = self.motion_dir
        self.events.init_motion_loader.params["root_name"] = self.root_link_name
        self.events.init_motion_loader.params["all_body_names"] = self.amp_all_body_names
        self.events.reset_from_motion.params["motion_dir"] = self.motion_dir
        self.events.reset_from_motion.params["root_name"] = self.root_link_name
        self.events.reset_from_motion.params["all_body_names"] = self.amp_all_body_names
        self.events.reset_from_motion.params["asset_cfg"].joint_names = list(self.joint_names)
        self.events.reset_from_motion.params["recovery_env_ratio"] = self.recovery_env_ratio
        self.events.reset_from_motion.params["recovery_name_patterns"] = self.recovery_name_patterns
        self.events.reset_from_motion.params["recovery_max_steps"] = self.recovery_max_steps
        self.events.init_motion_loader.params["recovery_motion_dir"] = self.recovery_motion_dir
        self.events.init_motion_loader.params["recovery_name_patterns"] = self.recovery_name_patterns
        self.events.recovery_assist_force.params["asset_cfg"].body_names = [self.root_link_name]
        self.events.update_recovery_status.params["success_height"] = self.recovery_success_height
        self.events.update_recovery_status.params["success_angle"] = self.recovery_success_angle
        self.events.update_recovery_status.params["success_hold_steps"] = self.recovery_success_hold_steps
        self.events.recovery_assist_force.params["force_start"] = self.assist_force_start
        self.events.recovery_assist_force.params["force_end"] = self.assist_force_end
        self.events.recovery_assist_force.params["anneal_steps"] = self.assist_anneal_steps

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = list(self.joint_names)
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = list(self.joint_names)
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = list(self.joint_names)
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = list(self.joint_names)

        self.observations.critic.body_pos_b.params["anchor_cfg"].body_names = [self.anchor_body_name]
        self.observations.critic.body_pos_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.critic.body_ori_b.params["anchor_cfg"].body_names = [self.anchor_body_name]
        self.observations.critic.body_ori_b.params["body_cfg"].body_names = list(self.amp_body_names)

        self.observations.amp_policy.body_pos_b.params["anchor_cfg"].body_names = [self.anchor_body_name]
        self.observations.amp_policy.body_pos_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp_policy.body_ori_b.params["anchor_cfg"].body_names = [self.anchor_body_name]
        self.observations.amp_policy.body_ori_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp_policy.body_lin_vel_b.params["anchor_cfg"].body_names = [self.anchor_body_name]
        self.observations.amp_policy.body_lin_vel_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp_policy.body_ang_vel_b.params["anchor_cfg"].body_names = [self.anchor_body_name]
        self.observations.amp_policy.body_ang_vel_b.params["body_cfg"].body_names = list(self.amp_body_names)

        self.observations.amp_expert.body_pos_b.params["anchor_cfg"].body_names = [self.anchor_body_name]
        self.observations.amp_expert.body_pos_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp_expert.body_ori_b.params["anchor_cfg"].body_names = [self.anchor_body_name]
        self.observations.amp_expert.body_ori_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp_expert.body_lin_vel_b.params["anchor_cfg"].body_names = [self.anchor_body_name]
        self.observations.amp_expert.body_lin_vel_b.params["body_cfg"].body_names = list(self.amp_body_names)
        self.observations.amp_expert.body_ang_vel_b.params["anchor_cfg"].body_names = [self.anchor_body_name]
        self.observations.amp_expert.body_ang_vel_b.params["body_cfg"].body_names = list(self.amp_body_names)

        self.rewards.track_anchor_linear_velocity.params["anchor_cfg"].body_names = [self.anchor_body_name]
        self.rewards.track_anchor_angular_velocity.params["anchor_cfg"].body_names = [self.anchor_body_name]
        self.rewards.body_ang_vel_xy_l2.params["body_cfg"].body_names = [self.anchor_body_name]
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.root_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [".*ankle_roll_link"]
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [".*ankle_roll_link"]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!.*ankle_roll_link$).+$",
        ]

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            r"^(?!.*ankle_roll_link$).+$",
        ]

        # These are the generic AMP locomotion weights. They remain robot-local so another
        # robot can use the same public environment with its own reward balance.
        self.rewards.is_terminated.weight = -200.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.joint_pos_limits.weight = -0.5
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.feet_slide.weight = -0.2
        self.rewards.undesired_contacts.weight = -1.0


@configclass
class G1AmpNoRecoveryEnvCfg(G1AmpEnvCfg):
    """G1 AMP environment trained only on regular motion clips."""

    recovery_env_ratio = 0.0
    assist_force_start = 0.0
    assist_force_end = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep recovery clips excluded from normal motion loading, while disabling
        # all recovery-only runtime events for this configuration.
        self.events.update_recovery_status = None
        self.events.recovery_assist_force = None
        self.terminations.recovery_failure = None
