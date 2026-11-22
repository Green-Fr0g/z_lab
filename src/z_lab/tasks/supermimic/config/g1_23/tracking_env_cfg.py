"""Bind the G1 23-DOF asset and reference motion to the SuperMimic environments."""

from pathlib import Path

from isaaclab.utils import configclass

from z_lab.assets.robots.g1_23 import (
    G1_23DOF_ANCHOR_LINK_NAME,
    G1_23DOF_DOF_ORDER,
    G1_23DOF_MIMIC_LINK_NAMES,
    G1_23DOF_ROOT_LINK_NAME,
    MIMIC_G1_23DOF_ACTION_SCALE,
    MIMIC_G1_23DOF_CFG,
)
from z_lab.tasks.supermimic.tracking_env_cfg import (
    GenCoordStateCfg,
    MaxCoordStateCfg,
    ObservationsCfg,
    SuperMimicGenCoordEnvCfg,
)

MOTIONS_DIR = Path(__file__).resolve().parent / "motions"


@configclass
class G1_23DOFSuperMimicGenCoordEnvCfg(SuperMimicGenCoordEnvCfg):
    """G1 23-DOF motion tracking with generalized-coordinate observations."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = MIMIC_G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.joint_names = G1_23DOF_DOF_ORDER
        self.actions.joint_pos.scale = MIMIC_G1_23DOF_ACTION_SCALE

        self.commands.motion.motion_file = [str(path) for path in sorted(MOTIONS_DIR.glob("*.npz")) if path.is_file()]
        self.commands.motion.anchor_body_name = G1_23DOF_ROOT_LINK_NAME
        self.commands.motion.body_names = G1_23DOF_MIMIC_LINK_NAMES
        self.commands.motion.joint_names = G1_23DOF_DOF_ORDER

        self.observations.actor.joint_pos.params["asset_cfg"].joint_names = G1_23DOF_DOF_ORDER
        self.observations.actor.joint_vel.params["asset_cfg"].joint_names = G1_23DOF_DOF_ORDER
        if isinstance(self.observations.full_state, GenCoordStateCfg):
            self.observations.full_state.state.params["asset_cfg"].joint_names = G1_23DOF_DOF_ORDER

        self.rewards.motion_global_anchor_pos.weight = 0.5
        self.rewards.motion_global_anchor_ori.weight = 0.5
        self.rewards.motion_body_pos.weight = 1.0
        self.rewards.motion_body_ori.weight = 1.0
        self.rewards.motion_body_lin_vel.weight = 1.0
        self.rewards.motion_body_ang_vel.weight = 1.0
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.joint_limit.weight = -2.0

        self.events.foot_friction.params["asset_cfg"].body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.events.base_com.params["asset_cfg"].body_names = [G1_23DOF_ANCHOR_LINK_NAME]


@configclass
class G1_23DOFSuperMimicMaxCoordEnvCfg(G1_23DOFSuperMimicGenCoordEnvCfg):
    """G1 23-DOF motion tracking with maximal-coordinate observations."""

    observations: ObservationsCfg = ObservationsCfg(full_state=MaxCoordStateCfg())
