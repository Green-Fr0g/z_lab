"""Bind the G1 23-DOF asset, state order, expert motions, and reward weights."""

from isaaclab.utils import configclass

from z_lab.assets.robots.g1_23 import (
    G1_23DOF_ACTION_SCALE,
    G1_23DOF_ANCHOR_LINK_NAME,
    G1_23DOF_CFG,
    G1_23DOF_DOF_ORDER,
    G1_23DOF_MIMIC_LINK_NAMES,
    G1_23DOF_ROOT_LINK_NAME,
)
from z_lab.tasks.amp.config.g1_23.amp_env_cfg import MOTION_DIR, RECOVERY_CLIP_NAME
from z_lab.tasks.superamp.superamp_env_cfg import MaxCoordStateCfg, ObservationsCfg, SuperAmpGenCoordEnvCfg


@configclass
class G1_23DOFSuperAmpGenCoordEnvCfg(SuperAmpGenCoordEnvCfg):
    """G1 23-DOF velocity tracking with generalized-coordinate observations."""

    joint_names = G1_23DOF_DOF_ORDER
    full_state_body_names = G1_23DOF_MIMIC_LINK_NAMES
    anchor_link_name = G1_23DOF_ANCHOR_LINK_NAME
    root_link_name = G1_23DOF_ROOT_LINK_NAME
    foot_links = ["left_ankle_roll_link", "right_ankle_roll_link"]
    amp_action_scale = G1_23DOF_ACTION_SCALE

    # The bundled AMP clips were exported with center-of-mass linear velocities.
    expert_linear_velocity_at_com: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

        # ------------------------------scene------------------------------
        self.scene.robot = G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------actions------------------------------
        self.actions.joint_pos.joint_names = self.joint_names
        self.actions.joint_pos.scale = self.amp_action_scale

        # ------------------------------observations------------------------------

        # actor related
        self.observations.actor.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.actor.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # world model state
        state_asset = self.observations.full_state.state.params["asset_cfg"]
        if isinstance(self.observations.full_state, MaxCoordStateCfg):
            state_asset.body_names = self.full_state_body_names
        else:
            state_asset.joint_names = self.joint_names

        # ------------------------------rewards------------------------------

        # track related
        self.rewards.track_lin_vel_xy.weight = 1.0
        self.rewards.track_ang_vel_z.weight = 1.0

        # joint related
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.action_rate_l2.weight = -0.01

        # ------------------------------params------------------------------
        self.rewards.track_lin_vel_xy.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.rewards.track_ang_vel_z.params["anchor_cfg"].body_names = [self.anchor_link_name]
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------events------------------------------

        # expert motion reset
        self.events.reset_from_expert.params["motion_file"] = str(MOTION_DIR)
        self.events.reset_from_expert.params["root_name"] = self.root_link_name
        self.events.reset_from_expert.params["exclude_motion_names"] = (RECOVERY_CLIP_NAME,)

        # startup randomization
        self.events.foot_friction.params["asset_cfg"].body_names = self.foot_links
        self.events.base_com.params["asset_cfg"].body_names = [self.anchor_link_name]


@configclass
class G1_23DOFSuperAmpMaxCoordEnvCfg(G1_23DOFSuperAmpGenCoordEnvCfg):
    """G1 23-DOF velocity tracking with maximal-coordinate observations."""

    observations: ObservationsCfg = ObservationsCfg(full_state=MaxCoordStateCfg())
