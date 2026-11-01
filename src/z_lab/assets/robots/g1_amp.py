"""AMP-specific G1 actuator and initial-state configuration."""

import re

from isaaclab.actuators import ImplicitActuatorCfg

from .g1 import G1_CFG, G1_DOF_ORDER

_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0


def _reflected_armature(rotor_inertias: tuple[float, ...], gears: tuple[float, ...]) -> float:
    return sum(inertia * gear**2 for inertia, gear in zip(rotor_inertias, gears, strict=True))


_ARMATURE_5020 = _reflected_armature((0.139e-4, 0.017e-4, 0.169e-4), (1.0, 1 + 46 / 18, 1 + 56 / 16))
_ARMATURE_7520_14 = _reflected_armature((0.489e-4, 0.098e-4, 0.533e-4), (1.0, 4.5, 1 + 48 / 22))
_ARMATURE_7520_22 = _reflected_armature((0.489e-4, 0.109e-4, 0.738e-4), (1.0, 4.5, 5.0))
_ARMATURE_5010_16 = 0.0021812


def _pd(armature: float, multiplicity: int = 1) -> tuple[float, float, float]:
    inertia = armature * multiplicity
    stiffness = inertia * _NATURAL_FREQ**2
    damping = 2.0 * _DAMPING_RATIO * inertia * _NATURAL_FREQ
    return stiffness, damping, inertia


_KP_5020, _KD_5020, _ = _pd(_ARMATURE_5020)
_KP_7520_14, _KD_7520_14, _ = _pd(_ARMATURE_7520_14)
_KP_7520_22, _KD_7520_22, _ = _pd(_ARMATURE_7520_22)
_KP_5010_16, _KD_5010_16, _ = _pd(_ARMATURE_5010_16)
_KP_WAIST_ANKLE, _KD_WAIST_ANKLE, _ARMATURE_WAIST_ANKLE = _pd(_ARMATURE_5020, multiplicity=2)

G1_AMP_CFG = G1_CFG.replace(
    init_state=G1_CFG.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", "waist_yaw_joint"],
            joint_effort_limit=88.0,
            actuator_velocity_limit=32.0,
            stiffness=_KP_7520_14,
            damping=_KD_7520_14,
            armature=_ARMATURE_7520_14,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"],
            joint_effort_limit=139.0,
            actuator_velocity_limit=20.0,
            stiffness=_KP_7520_22,
            damping=_KD_7520_22,
            armature=_ARMATURE_7520_22,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint"],
            joint_effort_limit=25.0,
            actuator_velocity_limit=37.0,
            stiffness=_KP_5020,
            damping=_KD_5020,
            armature=_ARMATURE_5020,
        ),
        "N5020-16-waist-ankle": ImplicitActuatorCfg(
            joint_names_expr=["waist_(pitch|roll)_joint", ".*_ankle_.*"],
            joint_effort_limit=50.0,
            actuator_velocity_limit=37.0,
            stiffness=_KP_WAIST_ANKLE,
            damping=_KD_WAIST_ANKLE,
            armature=_ARMATURE_WAIST_ANKLE,
        ),
        "N5010-16": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            joint_effort_limit=10.0,
            actuator_velocity_limit=22.0,
            stiffness=_KP_5010_16,
            damping=_KD_5010_16,
            armature=_ARMATURE_5010_16,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)


def _resolve_actuator_value(cfg, joint_name: str, attribute: str) -> float | None:
    """Resolve an actuator scalar or first matching regex value for one joint."""
    values = getattr(cfg, attribute, None)
    if values is None:
        return None
    if not isinstance(values, dict):
        return float(values)
    for pattern, value in values.items():
        if re.fullmatch(pattern, joint_name):
            return float(value)
    return 0.0


def _amp_action_scale() -> dict[str, float]:
    scales = {}
    for joint_name in G1_DOF_ORDER:
        actuator = next(
            (
                cfg
                for cfg in G1_AMP_CFG.actuators.values()
                if any(re.fullmatch(pattern, joint_name) for pattern in cfg.joint_names_expr)
            ),
            None,
        )
        if actuator is None:
            continue
        effort = _resolve_actuator_value(actuator, joint_name, "joint_effort_limit")
        stiffness = _resolve_actuator_value(actuator, joint_name, "stiffness")
        if effort and stiffness:
            scales[joint_name] = 0.25 * effort / stiffness
    return scales


G1_AMP_ACTION_SCALE = _amp_action_scale()
