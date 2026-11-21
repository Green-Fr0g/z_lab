"""Bind the G1 asset and reference motion to the SuperMimic environments."""

from isaaclab.utils import configclass

from z_lab.assets.robots.g1 import (
    G1_DOF_ORDER,
    G1_MIMIC_LINK_NAMES,
    G1_ROOT_LINK_NAME,
    MIMIC_G1_ACTION_SCALE,
    MIMIC_G1_CFG,
)
from z_lab.tasks.mimic.config.g1.tracking_env_cfg import MOTION_FILE
from z_lab.tasks.supermimic.tracking_env_cfg import (
    GenCoordStateCfg,
    MaxCoordStateCfg,
    ObservationsCfg,
    SuperMimicGenCoordEnvCfg,
)


@configclass
class G1SuperMimicGenCoordEnvCfg(SuperMimicGenCoordEnvCfg):
    """G1 motion tracking with generalized-coordinate observations."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.robot = MIMIC_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.joint_names = G1_DOF_ORDER
        self.actions.joint_pos.scale = MIMIC_G1_ACTION_SCALE

        self.commands.motion.motion_file = str(MOTION_FILE)
        self.commands.motion.anchor_body_name = G1_ROOT_LINK_NAME
        self.commands.motion.body_names = G1_MIMIC_LINK_NAMES
        self.commands.motion.joint_names = G1_DOF_ORDER

        self.observations.actor.joint_pos.params["asset_cfg"].joint_names = G1_DOF_ORDER
        self.observations.actor.joint_vel.params["asset_cfg"].joint_names = G1_DOF_ORDER
        if isinstance(self.observations.full_state, GenCoordStateCfg):
            self.observations.full_state.state.params["asset_cfg"].joint_names = G1_DOF_ORDER

        self.rewards.motion_global_anchor_pos.weight = 0.5
        self.rewards.motion_global_anchor_ori.weight = 0.5
        self.rewards.motion_body_pos.weight = 1.0
        self.rewards.motion_body_ori.weight = 1.0
        self.rewards.motion_body_lin_vel.weight = 1.0
        self.rewards.motion_body_ang_vel.weight = 1.0
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.joint_limit.weight = -2.0

        self.events.foot_friction.params["asset_cfg"].body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.events.base_com.params["asset_cfg"].body_names = ["torso_link"]


@configclass
class G1SuperMimicMaxCoordEnvCfg(G1SuperMimicGenCoordEnvCfg):
    """G1 motion tracking with maximal-coordinate observations."""

    observations: ObservationsCfg = ObservationsCfg(full_state=MaxCoordStateCfg())
