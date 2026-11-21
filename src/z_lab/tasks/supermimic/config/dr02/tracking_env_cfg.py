"""Bind the DR02 asset and reference motion to the SuperMimic environments."""

from isaaclab.utils import configclass

from z_lab.assets.robots.dr02 import (
    DR02_ACTION_SCALE,
    DR02_ANCHOR_LINK_NAME,
    DR02_CFG,
    DR02_DOF_ORDER,
    DR02_MIMIC_LINK_NAMES,
    DR02_ROOT_LINK_NAME,
)
from z_lab.tasks.mimic.config.dr02.tracking_env_cfg import MOTION_FILE
from z_lab.tasks.supermimic.tracking_env_cfg import (
    GenCoordStateCfg,
    MaxCoordStateCfg,
    ObservationsCfg,
    SuperMimicGenCoordEnvCfg,
)


@configclass
class DR02SuperMimicGenCoordEnvCfg(SuperMimicGenCoordEnvCfg):
    """DR02 motion tracking with generalized-coordinate observations."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = DR02_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.joint_names = DR02_DOF_ORDER
        self.actions.joint_pos.scale = DR02_ACTION_SCALE

        self.commands.motion.motion_file = str(MOTION_FILE)
        self.commands.motion.anchor_body_name = DR02_ROOT_LINK_NAME
        # Track the last wrist link so the analytical IK includes all three wrist joints.
        self.commands.motion.body_names = [
            name.replace("wrist_y_link", "wrist_x_link") for name in DR02_MIMIC_LINK_NAMES
        ]
        self.commands.motion.joint_names = DR02_DOF_ORDER

        self.observations.actor.joint_pos.params["asset_cfg"].joint_names = DR02_DOF_ORDER
        self.observations.actor.joint_vel.params["asset_cfg"].joint_names = DR02_DOF_ORDER
        if isinstance(self.observations.full_state, GenCoordStateCfg):
            self.observations.full_state.state.params["asset_cfg"].joint_names = DR02_DOF_ORDER

        self.rewards.motion_global_anchor_pos.weight = 0.5
        self.rewards.motion_global_anchor_ori.weight = 0.5
        self.rewards.motion_body_pos.weight = 1.0
        self.rewards.motion_body_ori.weight = 1.0
        self.rewards.motion_body_lin_vel.weight = 1.0
        self.rewards.motion_body_ang_vel.weight = 1.0
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.joint_limit.weight = -2.0
        self.rewards.joint_limit.params["asset_cfg"].joint_names = DR02_DOF_ORDER

        self.events.foot_friction.params["asset_cfg"].body_names = ["left_ankle_x_link", "right_ankle_x_link"]
        self.events.base_com.params["asset_cfg"].body_names = [DR02_ANCHOR_LINK_NAME]


@configclass
class DR02SuperMimicMaxCoordEnvCfg(DR02SuperMimicGenCoordEnvCfg):
    """DR02 motion tracking with maximal-coordinate observations."""

    observations: ObservationsCfg = ObservationsCfg(full_state=MaxCoordStateCfg())
