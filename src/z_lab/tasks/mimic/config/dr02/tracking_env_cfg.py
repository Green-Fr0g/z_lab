"""DR02 motion-mimic environment configuration."""

from pathlib import Path

from isaaclab.utils import configclass

from z_lab.assets.robots.dr02 import (
    DR02_ACTION_SCALE,
    DR02_ANCHOR_LINK_NAME,
    DR02_CFG,
    DR02_DOF_ORDER,
)
from z_lab.tasks.mimic.tracking_env_cfg import TrackingEnvCfg

MOTION_FILE = Path(__file__).resolve().parent / "motions" / "dance.npz"


@configclass
class DR02MimicEnvCfg(TrackingEnvCfg):
    """DR02 motion-mimic task."""

    anchor_body_name = DR02_ANCHOR_LINK_NAME
    body_names = [
        "base_link",
        "body",
        "left_hip_x_link",
        "left_knee_link",
        "left_ankle_x_link",
        "right_hip_x_link",
        "right_knee_link",
        "right_ankle_x_link",
        "left_shoulder_x_link",
        "left_elbow_link",
        "left_wrist_y_link",
        "right_shoulder_x_link",
        "right_elbow_link",
        "right_wrist_y_link",
    ]
    limb_end_body_names = [
        "left_ankle_x_link",
        "right_ankle_x_link",
        "left_wrist_y_link",
        "right_wrist_y_link",
    ]

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = DR02_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.joint_names = list(DR02_DOF_ORDER)
        self.actions.joint_pos.scale = DR02_ACTION_SCALE
        self.commands.motion.motion_file = str(MOTION_FILE)
        self.commands.motion.anchor_body_name = self.anchor_body_name
        self.commands.motion.body_names = self.body_names
        # The USD also contains two neck joints which are not policy actuators; the
        # generic default-position randomizer indexes action-local offsets, so it
        # cannot be applied to this partial action set.
        self.events.add_joint_default_pos = None

        self.rewards.motion_global_anchor_pos.weight = 0.5
        self.rewards.motion_global_anchor_ori.weight = 0.5
        self.rewards.motion_body_pos.weight = 1.0
        self.rewards.motion_body_ori.weight = 1.0
        self.rewards.motion_body_lin_vel.weight = 1.0
        self.rewards.motion_body_ang_vel.weight = 1.0
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.joint_limit.weight = -10.0
        self.rewards.undesired_contacts.weight = -0.1

        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!.*ankle_x_link$)(?!.*wrist_y_link$).+$"
        ]
        self.terminations.ee_body_pos.params["body_names"] = self.limb_end_body_names


@configclass
class DR02MimicWoStateEstimationEnvCfg(DR02MimicEnvCfg):
    """DR02 Mimic task without selected state-estimation observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
