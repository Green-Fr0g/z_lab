"""DR02-Pro asset configuration for Isaac Lab 3.0.

The local USD has 31 movable joints, configured in one actuator group.
``DR02_DOF_ORDER`` contains the 29 policy joints (including waist and wrists).
Select ``DR02_DOF_ORDER`` explicitly in policy action terms.

All joints use implicit PD drives handled by the physics solver.
"""

import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from z_lab.assets import Z_LAB_ASSETS_DIR

# ---------------------------------------------------------------------------
# DR02 (DR02-Pro)
# ---------------------------------------------------------------------------

DR02_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Z_LAB_ASSETS_DIR / "models" / "dr02" / "DR02-pro.usd"),
        fix_root_link=False,
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100 * 57.1,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.92),
        joint_pos={
            # Additional joints in the full USD start at their neutral pose.
            "waist_[xy]_joint|.*_wrist_[xyz]_joint|neck_[zy]_joint": 0.0,
            # legs
            "left_hip_y_joint": -0.1,
            "left_hip_x_joint": 0.0,
            "left_hip_z_joint": 0.0,
            "left_knee_joint": 0.2,
            "left_ankle_y_joint": -0.1,
            "left_ankle_x_joint": 0.0,
            "right_hip_y_joint": -0.1,
            "right_hip_x_joint": 0.0,
            "right_hip_z_joint": 0.0,
            "right_knee_joint": 0.2,
            "right_ankle_y_joint": -0.1,
            "right_ankle_x_joint": 0.0,
            # waist
            "waist_z_joint": 0.0,
            # arms
            "left_shoulder_y_joint": 0.0,
            "left_shoulder_x_joint": 0.15,
            "left_shoulder_z_joint": 0.0,
            "left_elbow_joint": 1.35,
            "right_shoulder_y_joint": 0.0,
            "right_shoulder_x_joint": -0.15,
            "right_shoulder_z_joint": 0.0,
            "right_elbow_joint": 1.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        # All 31 joints, with parameters selected by joint-name patterns.
        "joints": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_[xyz]_joint",
                ".*_knee_joint",
                ".*_ankle_[xy]_joint",
                "waist_[xyz]_joint",
                ".*_shoulder_[xyz]_joint",
                ".*_elbow_joint",
                ".*_wrist_[xyz]_joint",
                "neck_[zy]_joint",
            ],
            joint_effort_limit={
                ".*_hip_y_joint|.*_hip_x_joint|.*_knee_joint": 330.0,
                ".*_hip_z_joint|.*_ankle_y_joint|waist_z_joint|.*_shoulder_[xyz]_joint|.*_elbow_joint": 105.0,
                ".*_ankle_x_joint": 35.0,
                # Additional joints retain the effort limits authored in the USD.
                "waist_x_joint": 137.0,
                "waist_y_joint": 363.0,
                ".*_wrist_[xyz]_joint|neck_[zy]_joint": 50.0,
            },
            actuator_velocity_limit={
                ".*_hip_y_joint|.*_hip_x_joint|.*_knee_joint": 18.0,
                ".*_hip_z_joint|.*_ankle_y_joint|waist_z_joint|.*_shoulder_[xyz]_joint|.*_elbow_joint": 17.38,
                ".*_ankle_x_joint": 20.76,
                # USD angular velocity limits converted from degrees/s to rad/s.
                "waist_x_joint": 19.38,
                "waist_y_joint": 20.0,
                ".*_wrist_[xyz]_joint|neck_[zy]_joint": 23.76,
            },
            joint_velocity_limit={
                ".*_hip_[xyz]_joint|.*_knee_joint|.*_ankle_[xy]_joint": 1e7,
                "waist_z_joint|.*_shoulder_[xyz]_joint|.*_elbow_joint": 1e7,
                "waist_x_joint": 19.38,
                "waist_y_joint": 20.0,
                ".*_wrist_[xyz]_joint|neck_[zy]_joint": 23.76,
            },
            stiffness={
                ".*_hip_y_joint|.*_hip_x_joint|.*_knee_joint": 250.0,
                ".*_hip_z_joint": 180.0,
                ".*_ankle_y_joint|.*_shoulder_y_joint|.*_shoulder_x_joint|.*_shoulder_z_joint|.*_elbow_joint": 100.0,
                "waist_[xyz]_joint": 150.0,
                ".*_ankle_x_joint": 40.0,
                ".*_wrist_[xyz]_joint": 80.0,
                "neck_[zy]_joint": 80.0,
            },
            damping={
                ".*_hip_y_joint|.*_hip_x_joint|.*_knee_joint": 6.0,
                ".*_hip_z_joint": 4.0,
                ".*_ankle_y_joint|.*_shoulder_y_joint|.*_shoulder_x_joint|.*_shoulder_z_joint|.*_elbow_joint": 2.5,
                "waist_[xyz]_joint": 3.0,
                ".*_ankle_x_joint": 1.0,
                ".*_wrist_[xyz]_joint": 2.0,
                "neck_[zy]_joint": 2.0,
            },
        ),
    },
)

DR02_DOF_ORDER = (
    "left_hip_y_joint",
    "left_hip_x_joint",
    "left_hip_z_joint",
    "left_knee_joint",
    "left_ankle_y_joint",
    "left_ankle_x_joint",
    "right_hip_y_joint",
    "right_hip_x_joint",
    "right_hip_z_joint",
    "right_knee_joint",
    "right_ankle_y_joint",
    "right_ankle_x_joint",
    "waist_z_joint",
    "waist_x_joint",
    "waist_y_joint",
    "left_shoulder_y_joint",
    "left_shoulder_x_joint",
    "left_shoulder_z_joint",
    "left_elbow_joint",
    "left_wrist_z_joint",
    "left_wrist_y_joint",
    "left_wrist_x_joint",
    "right_shoulder_y_joint",
    "right_shoulder_x_joint",
    "right_shoulder_z_joint",
    "right_elbow_joint",
    "right_wrist_z_joint",
    "right_wrist_y_joint",
    "right_wrist_x_joint",
)
DR02_AMP_DOF_ORDER = DR02_DOF_ORDER

DR02_ROOT_LINK_NAME = "base_link"
DR02_ANCHOR_LINK_NAME = "body"
DR02_AMP_LINK_NAMES = (
    "base_link",
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
)


def _dr02_actuator_value(joint_name: str, attribute: str) -> float | None:
    """Resolve one actuator parameter for a DR02 joint."""
    for actuator_cfg in DR02_CFG.actuators.values():
        if not any(re.fullmatch(pattern, joint_name) for pattern in actuator_cfg.joint_names_expr):
            continue
        values = getattr(actuator_cfg, attribute, None)
        if values is None:
            return None
        if not isinstance(values, dict):
            return float(values)
        for pattern, value in values.items():
            if re.fullmatch(pattern, joint_name):
                return float(value)
        return 0.0
    return None


def _dr02_action_scale() -> dict[str, float]:
    """Map each policy joint to ``0.25 * effort_limit / stiffness``."""
    scales = {}
    for joint_name in DR02_DOF_ORDER:
        effort_limit = _dr02_actuator_value(joint_name, "joint_effort_limit")
        stiffness = _dr02_actuator_value(joint_name, "stiffness")
        if effort_limit and stiffness:
            scales[joint_name] = 0.25 * effort_limit / stiffness
    return scales


DR02_ACTION_SCALE = _dr02_action_scale()
