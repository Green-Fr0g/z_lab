from copy import deepcopy
from pathlib import Path

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from z_lab.assets.robots.g1 import G1_AMP_DOF_ORDER, G1_AMP_LINK_NAMES, G1_ANCHOR_LINK_NAME, G1_CFG, G1_ROOT_LINK_NAME
from z_lab.tasks.amp.amp_env_cfg import AmpEnvCfg


@configclass
class UnitreeG1AmpEnvCfg(AmpEnvCfg):
    """AMP task for the 29-DoF Unitree G1 on flat ground."""

    # links that carry the discriminators' body features, chosen to cover the whole silhouette
    amp_body_names: tuple[str, ...] = G1_AMP_LINK_NAMES
    amp_anchor_name: str = G1_ANCHOR_LINK_NAME
    amp_joint_names: tuple[str, ...] = G1_AMP_DOF_ORDER

    def __post_init__(self) -> None:
        super().__post_init__()

        # ------------------------------ scene ------------------------------
        self.scene.robot = deepcopy(G1_CFG)
        self.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"

        # ------------------------------ actions ------------------------------
        self.actions.joint_position.joint_names = list(G1_AMP_DOF_ORDER)

        # ------------------------------ commands ------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.limit_lin_vel_x = (-1.0, 1.5)
        self.commands.base_velocity.ranges.limit_lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.limit_ang_vel_z = (-1.0, 1.0)

        # ------------------------------ events ------------------------------
        self.events.init_amp_motion.params["motion_dir"] = str(Path(__file__).resolve().parent / "motions")
        self.events.init_amp_motion.params["root_name"] = G1_ROOT_LINK_NAME
        # a fifth of the batch keeps training getting up instead of only steady walking
        self.events.init_delayed_termination.params["delay_reset_env_ratio"] = 0.2
        self.events.init_delayed_termination.params["max_delay_steps"] = 50

        # ------------------------------ rewards ------------------------------
        # Tracking and recovery rewards.
        self.rewards.track_anchor_linear_velocity.weight = 1.0
        self.rewards.track_anchor_angular_velocity.weight = 0.5
        self.rewards.track_root_height.weight = 1.0
        self.rewards.body_ang_vel_xy_l2.weight = 0.5

        # Termination and joint/action penalties.
        self.rewards.is_terminated.weight = -200.0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.action_rate_l2.weight = -0.01

        # ------------------------------ reward parameters ------------------------------
        self.rewards.track_anchor_linear_velocity.params["anchor_cfg"] = SceneEntityCfg(
            "robot", body_names=[self.amp_anchor_name]
        )
        self.rewards.track_anchor_angular_velocity.params["anchor_cfg"] = SceneEntityCfg(
            "robot", body_names=[self.amp_anchor_name]
        )
        self.rewards.body_ang_vel_xy_l2.params["body_cfg"] = SceneEntityCfg("robot", body_names=[G1_ROOT_LINK_NAME])


@configclass
class UnitreeG1AmpNoRecoveryEnvCfg(UnitreeG1AmpEnvCfg):
    """Locomotion-only AMP: normal motion resets and immediate reset on falls."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.init_delayed_termination.params["delay_reset_env_ratio"] = 0.0
        self.events.init_delayed_termination.params["max_delay_steps"] = 0
        self.events.reset_from_motion.params["use_recovery"] = False
        self.rewards.track_root_height.weight = 0.0
