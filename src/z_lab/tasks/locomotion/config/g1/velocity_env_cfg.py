from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from z_lab.assets.robots.g1 import G1_CFG, G1_DOF_ORDER,G1_ROOT_LINK_NAME

from z_lab.tasks.locomotion import mdp
from z_lab.tasks.locomotion.locomotion_env_cfg import LocomotionEnvCfg


@configclass
class UnitreeG1VelocityEnvCfg(LocomotionEnvCfg):
    arm_joints = [".*_shoulder_.*_joint",".*_elbow_joint",".*_wrist_.*"]
    waist_joints = ["waist.*"]
    hip_joints = [".*_hip_roll_joint", ".*_hip_yaw_joint"]
    foot_links = [".*ankle_roll.*"]

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------scene------------------------------
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + G1_ROOT_LINK_NAME

        # ------------------------------actions------------------------------
        self.actions.joint_position.joint_names=list(G1_DOF_ORDER)

        # ------------------------------rewards------------------------------

        #track related
        self.rewards.track_lin_vel_xy.weight = 1.0
        self.rewards.track_ang_vel_z.weight = 0.8

        #root related
        self.rewards.base_height.weight = -10.0
        self.rewards.base_height.params["target_height"] = 0.78
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.alive.weight = 0.15
        self.rewards.base_linear_velocity.weight = -2.0
        self.rewards.base_angular_velocity.weight = -0.05
        self.rewards.energy.weight = -2e-5

        #joint related
        self.rewards.joint_vel.weight = -0.001
        self.rewards.joint_acc.weight = -2.5e-7
        self.rewards.action_rate.weight = -0.05
        self.rewards.dof_pos_limits.weight = -5.0
        self.rewards.joint_deviation_arms.weight = -0.15
        self.rewards.joint_deviation_waists.weight = -1.0
        self.rewards.joint_deviation_legs.weight = -0.8

        #feet related
        self.rewards.gait.weight = 0.5
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_clearance.weight = 1.0
        self.rewards.undesired_contacts.weight = -1.0

        # ------------------------------params------------------------------
        self.rewards.joint_deviation_arms.params["asset_cfg"].joint_names = self.arm_joints
        self.rewards.joint_deviation_waists.params["asset_cfg"].joint_names = self.waist_joints
        self.rewards.joint_deviation_legs.params["asset_cfg"].joint_names = self.hip_joints
        self.rewards.gait.params["sensor_cfg"].body_names = self.foot_links
        self.rewards.feet_slide.params["asset_cfg"].body_names = self.foot_links
        self.rewards.feet_slide.params["sensor_cfg"].body_names = self.foot_links
        self.rewards.feet_clearance.params["asset_cfg"].body_names = self.foot_links
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["(?!.*ankle).*"]

        # ------------------------------commands------------------------------
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        )
        self.commands.base_velocity.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
        )

        self.scene.terrain.terrain_type = "plane"
        self.curriculum.terrain_levels = None
