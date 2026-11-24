"""Build SuperMimic world models from Isaac Lab task state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from supermimic.kinematics import AnalyticIK, ForwardKinematics
from supermimic.models import Dynamics
from supermimic.representations import GenCoordRepresentation, MaxCoordRepresentation, StateRepresentation

if TYPE_CHECKING:
    from supermimic.envs.isaaclab_adapter import IsaacLabSuperMimicAdapter


@dataclass(frozen=True)
class WorldModel:
    """Dynamics and the task dimensions used to build it and its consumers."""

    representation: StateRepresentation
    dynamics: Dynamics
    num_goals: int
    proprio_dim: int
    sim_context_dim: int


def build_world_model_from_state(
    adapter: IsaacLabSuperMimicAdapter, representation_name: str, hidden_dims: tuple[int, ...]
) -> WorldModel:
    """Build the world model from an initialized task without resetting it."""
    env = adapter.env
    observations = adapter.observations()
    adapter.validate_full_state_layout(representation_name, observations["full_state"].shape[-1])
    num_joints = adapter.num_actions
    default_joint_pos = adapter.default_joint_pos()
    joint_limits = adapter.soft_joint_pos_limits()
    robot_model = adapter.load_robot_model()
    if representation_name == "gencoord":
        representation: StateRepresentation = GenCoordRepresentation(
            num_joints, default_joint_pos, env.step_dt, env.cfg.decimation
        )
        fk = ForwardKinematics(robot_model, dtype=default_joint_pos.dtype, device=adapter.device)
        representation.bind(forward_kinematics=fk, soft_joint_pos_limits=joint_limits)
    else:
        num_bodies = len(robot_model.body_names)
        representation = MaxCoordRepresentation(num_bodies, default_joint_pos, env.step_dt, env.cfg.decimation)
        ik = AnalyticIK(robot_model, device=adapter.device, dtype=default_joint_pos.dtype)
        representation.bind(analytic_ik=ik, soft_joint_pos_limits=joint_limits)
    representation = representation.to(adapter.device)
    sim_context_dim = adapter.sim_context().shape[-1]
    representation.default_joint_vel.copy_(adapter.default_joint_vel())
    representation.bind_encoder_bias_context(*adapter.encoder_bias_context(sim_context_dim))
    proprio_dim = adapter.validate_proprio_layout()
    if observations["actor"].shape[-1] != proprio_dim or proprio_dim != representation.proprio_obs_dim(num_joints):
        raise ValueError("Task actor observation width does not match model proprioception")
    action_scale, action_offset = adapter.action_scale_offset()
    dynamics = Dynamics(representation, num_joints, sim_context_dim, action_scale, action_offset, hidden_dims).to(
        adapter.device
    )
    return WorldModel(representation, dynamics, observations["goal"].shape[-1], proprio_dim, sim_context_dim)
