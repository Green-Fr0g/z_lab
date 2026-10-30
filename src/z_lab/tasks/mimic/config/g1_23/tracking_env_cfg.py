"""G1 23-DOF motion-mimic environment configuration."""

from pathlib import Path

from isaaclab.utils import configclass

from z_lab.assets.robots.g1_23 import (
    G1_23DOF_ANCHOR_LINK_NAME,
    G1_23DOF_DOF_ORDER,
    MIMIC_G1_23DOF_ACTION_SCALE,
    MIMIC_G1_23DOF_CFG,
)
from z_lab.tasks.mimic.tracking_env_cfg import TrackingEnvCfg

MOTION_FILE = Path(__file__).resolve().parent / "motions" / "dance.npz"


@configclass
class G1_23DOFMimicEnvCfg(TrackingEnvCfg):
    """G1 23-DOF motion-mimic task."""

    anchor_body_name = G1_23DOF_ANCHOR_LINK_NAME
    body_names = [
        "pelvis",
        "torso_link",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_roll_rubber_hand",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_roll_rubber_hand",
    ]
    limb_end_body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_roll_rubber_hand",
        "right_wrist_roll_rubber_hand",
    ]

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = MIMIC_G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.joint_names = list(G1_23DOF_DOF_ORDER)
        self.actions.joint_pos.scale = MIMIC_G1_23DOF_ACTION_SCALE
        self.commands.motion.motion_file = str(MOTION_FILE)
        self.commands.motion.anchor_body_name = self.anchor_body_name
        self.commands.motion.body_names = self.body_names
        self.commands.motion.joint_names = list(G1_23DOF_DOF_ORDER)
        self.events.add_joint_default_pos.params["asset_cfg"].joint_names = list(G1_23DOF_DOF_ORDER)

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = list(G1_23DOF_DOF_ORDER)
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = list(G1_23DOF_DOF_ORDER)
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = list(G1_23DOF_DOF_ORDER)
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = list(G1_23DOF_DOF_ORDER)

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
            r"^(?!.*ankle_roll_link$)(?!.*wrist_roll_rubber_hand$).+$"
        ]
        self.terminations.ee_body_pos.params["body_names"] = self.limb_end_body_names


@configclass
class G1_23DOFMimicWoStateEstimationEnvCfg(G1_23DOFMimicEnvCfg):
    """G1 23-DOF Mimic task without selected state-estimation observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
