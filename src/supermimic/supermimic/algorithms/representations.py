"""GenCoord and MaxCoord state representations for SuperMimic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from math import isfinite
from typing import Any

import torch
from torch import nn

from supermimic.utils.math import (
    matrix_to_quat,
    quat_apply,
    quat_apply_inverse,
    quat_integrate,
    quat_inv,
    quat_mul,
    quat_to_matrix,
    quat_to_rot6d,
    rodrigues_apply,
    rodrigues_precompute,
    rot6d_to_matrix,
)
from supermimic.utils.tracking_reward import tracking_reward

PROPRIO_OBS_TERMS = ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions")


def extract_actor_goal(wm_goal: torch.Tensor, num_goal_frames: int) -> torch.Tensor:
    """Remove the world-model-only relative-position part from a motion goal."""
    relative_pos_dim = 3 * num_goal_frames
    relative_ori_dim = 6 * num_goal_frames
    if num_goal_frames <= 0 or wm_goal.shape[-1] < relative_pos_dim + relative_ori_dim:
        raise ValueError("world-model goal width is incompatible with num_goal_frames")
    return torch.cat((wm_goal[..., : -relative_pos_dim - relative_ori_dim], wm_goal[..., -relative_ori_dim:]), dim=-1)


def goal_window_indices(
    motion: Any,
    motion_times: torch.Tensor,
    num_goal_frames: int,
    goal_frame_stride: int,
) -> torch.Tensor:
    """Return a future motion window clamped to each sampled clip's final frame."""
    if num_goal_frames <= 0 or goal_frame_stride <= 0:
        raise ValueError("goal frame count and stride must be positive")
    last_frame = motion.clip_end(motion_times) - 1
    offsets = torch.arange(num_goal_frames, device=motion_times.device) * goal_frame_stride
    return torch.minimum(motion_times[:, None] + offsets, last_frame[:, None])


def motion_goal_window(
    motion: Any,
    motion_times: torch.Tensor,
    num_goal_frames: int,
    goal_frame_stride: int,
    joint_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flatten future joint position and velocity targets frame by frame."""
    indices = goal_window_indices(motion, motion_times, num_goal_frames, goal_frame_stride)
    if joint_indices is None:
        joint_pos, joint_vel = motion.joint_pos[indices], motion.joint_vel[indices]
    else:
        joint_pos = motion.joint_pos[indices[..., None], joint_indices]
        joint_vel = motion.joint_vel[indices[..., None], joint_indices]
    content = torch.cat((joint_pos, joint_vel), dim=-1)
    return content.flatten(start_dim=1)


def motion_anchor_window(
    motion: Any,
    motion_times: torch.Tensor,
    num_goal_frames: int,
    goal_frame_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return future root poses in world coordinates for the same goal window."""
    indices = goal_window_indices(motion, motion_times, num_goal_frames, goal_frame_stride)
    return motion.root_pos_w[indices], motion.root_quat_w[indices]


def assemble_anchor_window(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express a batch of target poses relative to each current root pose."""
    num_envs, num_frames = target_pos.shape[:2]
    root_pos = root_pos[:, None, :].expand(num_envs, num_frames, 3)
    root_quat = root_quat[:, None, :].expand(num_envs, num_frames, 4)
    relative_pos = quat_apply_inverse(root_quat, target_pos - root_pos)
    relative_quat = quat_mul(quat_inv(root_quat), target_quat)
    return relative_pos.flatten(start_dim=1), quat_to_rot6d(relative_quat).flatten(start_dim=1)


class StateRepresentation(nn.Module, ABC):
    """Shared state and transition contract for world-model representations."""

    LOSS_WEIGHT_CONFIG: dict[str, str] = {
        "root_pos": "nominal_root_position_error",
        "root_ori": "nominal_root_orientation_error",
    }

    def __init__(
        self,
        num_joints: int,
        default_joint_pos: torch.Tensor,
        dt: float,
        decimation: int,
    ) -> None:
        super().__init__()
        if num_joints <= 0 or default_joint_pos.shape != (num_joints,):
            raise ValueError("default_joint_pos must contain one value per joint")
        if dt <= 0.0 or decimation <= 0:
            raise ValueError("dt and decimation must be positive")
        self.num_joints = num_joints
        self.dt = dt
        self.substep_coeff = (decimation + 1) / (2 * decimation)
        self.register_buffer("default_joint_pos", default_joint_pos.clone())
        self.register_buffer("default_joint_vel", torch.zeros_like(default_joint_pos), persistent=False)
        self.register_buffer("_proprio_noise_bounds", None, persistent=False)
        self._encoder_bias_context: tuple[slice, float, float] | None = None
        self.soft_joint_pos_limits: torch.Tensor | None = None

    @abstractmethod
    def represent(self, full_state: torch.Tensor) -> torch.Tensor:
        """Map a flattened physical state into this representation."""

    @abstractmethod
    def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
        """Extract world-frame root position and orientation."""

    @abstractmethod
    def apply_prediction(
        self,
        aux_state: torch.Tensor,
        rep_state: torch.Tensor,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate a predicted transition into auxiliary and representation states."""

    @abstractmethod
    def reconstruct_full_state(self, aux_state: torch.Tensor, rep_state: torch.Tensor) -> torch.Tensor:
        """Reconstruct the flattened physical state."""

    def bind(
        self,
        *,
        soft_joint_pos_limits: torch.Tensor | None = None,
        forward_kinematics: Any | None = None,
        analytic_ik: Any | None = None,
    ) -> None:
        """Attach environment- and robot-specific components after construction."""
        if soft_joint_pos_limits is not None:
            if soft_joint_pos_limits.shape != (self.num_joints, 2):
                raise ValueError("soft_joint_pos_limits must have shape [num_joints, 2]")
            self.soft_joint_pos_limits = soft_joint_pos_limits
        if forward_kinematics is not None:
            if not callable(forward_kinematics):
                raise TypeError("forward_kinematics must be callable")
            if not hasattr(self, "forward_kinematics"):
                raise TypeError("forward kinematics can only be bound to GenCoordRepresentation")
            self.forward_kinematics = forward_kinematics
        if analytic_ik is not None:
            if not callable(analytic_ik):
                raise TypeError("analytic_ik must be callable")
            if not hasattr(self, "analytic_ik"):
                raise TypeError("analytic IK can only be bound to MaxCoordRepresentation")
            self.analytic_ik = analytic_ik

    @abstractmethod
    def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
        """Return joint positions and velocities in policy joint order."""

    @abstractmethod
    def compute_wm_obs(
        self,
        rep_state: torch.Tensor,
        aux_state: torch.Tensor,
        joint_state: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Synthesize the policy's proprioceptive observation from model state."""

    def _validate_rep_state(self, rep_state: torch.Tensor) -> None:
        if rep_state.shape[-1] != self.rep_state_dim:
            raise ValueError(f"rep_state must end in {self.rep_state_dim} values")

    def configure_actor_observation(
        self,
        num_actions: int,
        noise_half_widths: Mapping[str, float],
    ) -> None:
        """Expand configured noise half-widths into the actor's proprioception layout."""
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if noise_half_widths.keys() != set(PROPRIO_OBS_TERMS):
            raise ValueError(f"Proprio noise must specify exactly these terms: {PROPRIO_OBS_TERMS}")
        for name, magnitude in noise_half_widths.items():
            if not isfinite(magnitude) or magnitude < 0.0:
                raise ValueError(f"Proprio noise half-width for {name!r} must be finite and non-negative")
        if not any(noise_half_widths.values()):
            self._proprio_noise_bounds = None
            return
        widths = (3, 3, self.num_joints, self.num_joints, num_actions)
        self._proprio_noise_bounds = torch.cat(
            [
                self.default_joint_pos.new_full((width,), noise_half_widths[name])
                for name, width in zip(PROPRIO_OBS_TERMS, widths, strict=True)
            ]
        )

    def bind_encoder_bias_context(self, columns: slice, scale: float, offset: float) -> None:
        """Decode the policy-ordered encoder bias from the normalized model context."""
        if columns.stop - columns.start != self.num_joints:
            raise ValueError("encoder-bias context must contain one value per joint")
        self._encoder_bias_context = (columns, scale, offset)

    def _joint_measurements(
        self, joint_state: torch.Tensor, sim_ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_pos, joint_vel = joint_state.split(self.num_joints, dim=-1)
        joint_pos_rel = joint_pos - self.default_joint_pos
        if self._encoder_bias_context is not None:
            columns, scale, offset = self._encoder_bias_context
            joint_pos_rel = joint_pos_rel + sim_ctx[..., columns] * scale + offset
        return joint_pos_rel, joint_vel - self.default_joint_vel

    def proprio_obs_dim(self, num_actions: int) -> int:
        return 6 + 2 * self.num_joints + num_actions

    def actor_obs_dim(
        self,
        num_actions: int,
        num_goals: int,
        history_length: int,
        num_goal_frames: int,
    ) -> int:
        if history_length <= 0 or num_goal_frames <= 0:
            raise ValueError("history_length and num_goal_frames must be positive")
        actor_goal_width = num_goals - 3 * num_goal_frames
        if actor_goal_width < 0:
            raise ValueError("num_goals is smaller than the world-model-only relative-position goal")
        return history_length * self.proprio_obs_dim(num_actions) + actor_goal_width

    def add_proprio_noise(self, observation: torch.Tensor) -> torch.Tensor:
        """Apply independent uniform noise when observation noise is configured."""
        if self._proprio_noise_bounds is None:
            return observation
        return observation + ((torch.rand_like(observation) * 2.0 - 1.0) * self._proprio_noise_bounds).detach()

    def compute_actor_sensor(
        self,
        rep_state: torch.Tensor,
        sim_ctx: torch.Tensor,
        aux_state: torch.Tensor,
        joint_state: torch.Tensor,
        last_action: torch.Tensor,
        noisy: bool = False,
    ) -> torch.Tensor:
        """Build proprioception from modeled state and the action that produced it."""
        proprio = self.compute_wm_obs(rep_state, aux_state, joint_state, sim_ctx)
        observation = torch.cat((proprio, last_action), dim=-1)
        return self.add_proprio_noise(observation) if noisy else observation

    @staticmethod
    def build_actor_obs(sensor_history: torch.Tensor, actor_goal: torch.Tensor) -> torch.Tensor:
        if sensor_history.ndim < 3:
            raise ValueError("sensor_history must have shape [batch, history, features]")
        return torch.cat((sensor_history.flatten(start_dim=1), actor_goal), dim=-1)

    def compute_wm_goal(
        self,
        motion: Any,
        motion_times: torch.Tensor,
        aux_state: torch.Tensor,
        num_goal_frames: int,
        goal_frame_stride: int,
        joint_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build world-model and actor motion goals using one shared window layout."""
        content = motion_goal_window(motion, motion_times, num_goal_frames, goal_frame_stride, joint_indices)
        target_pos, target_quat = motion_anchor_window(motion, motion_times, num_goal_frames, goal_frame_stride)
        relative_pos, relative_ori = assemble_anchor_window(
            aux_state[..., :3], aux_state[..., 3:7], target_pos, target_quat
        )
        return torch.cat((content, relative_pos, relative_ori), dim=-1), torch.cat((content, relative_ori), dim=-1)

    @abstractmethod
    def world_frame_bodies(
        self,
        aux_state: torch.Tensor,
        rep_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return world-frame body position, rotation matrix and velocities."""

    def compute_wm_reward_from_reference(
        self,
        next_rep_state: torch.Tensor,
        next_aux_state: torch.Tensor,
        next_joint_state: torch.Tensor,
        action: torch.Tensor,
        last_action: torch.Tensor,
        target_body_pos_w: torch.Tensor,
        target_body_quat_w: torch.Tensor,
        target_body_lin_vel_w: torch.Tensor,
        target_body_ang_vel_w: torch.Tensor,
        reward_weights: dict[str, float],
    ) -> torch.Tensor:
        """Compute the rollout reward against a reference body's world state."""
        if self.soft_joint_pos_limits is None:
            raise RuntimeError("soft joint position limits must be bound before computing rewards")
        if next_joint_state.shape[-1] != 2 * self.num_joints:
            raise ValueError(f"next_joint_state must end in {2 * self.num_joints} values")

        joint_pos = next_joint_state[..., : self.num_joints]
        return tracking_reward(
            next_aux_state,
            self.world_frame_bodies(next_aux_state, next_rep_state),
            action,
            last_action,
            joint_pos,
            (target_body_pos_w, target_body_quat_w, target_body_lin_vel_w, target_body_ang_vel_w),
            self.soft_joint_pos_limits,
            self.dt,
            reward_weights,
        )

    @abstractmethod
    def loss_blocks(self, rep_state: torch.Tensor, aux_state: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Return named channels used by the world-model loss."""


class GenCoordRepresentation(StateRepresentation):
    """Generalized-coordinate state and acceleration-level transition model."""

    LOSS_WEIGHT_CONFIG = {
        **StateRepresentation.LOSS_WEIGHT_CONFIG,
        "joint_pos": "nominal_joint_position_error",
        "joint_vel": "nominal_joint_velocity_error",
    }

    def __init__(self, num_joints: int, default_joint_pos: torch.Tensor, dt: float, decimation: int) -> None:
        super().__init__(num_joints, default_joint_pos, dt, decimation)
        self.full_state_dim = 13 + 2 * num_joints
        self.rep_state_dim = 10 + 2 * num_joints
        self.prediction_dim = 6 + num_joints
        self.forward_kinematics = None

    def represent(self, full_state: torch.Tensor) -> torch.Tensor:
        """Map xyzw full state to root-frame velocities and generalized state."""
        self._validate_full_state(full_state)
        joint_pos = full_state[..., 7 : 7 + self.num_joints]
        velocity_start = 7 + self.num_joints
        root_lin_vel = full_state[..., velocity_start : velocity_start + 3]
        root_ang_vel = full_state[..., velocity_start + 3 : velocity_start + 6]
        joint_vel = full_state[..., velocity_start + 6 : velocity_start + 6 + self.num_joints]
        root_quat = full_state[..., 3:7]
        world_down = torch.zeros_like(root_lin_vel)
        world_down[..., 2] = -1.0
        return torch.cat(
            (
                full_state[..., 2:3],
                quat_apply_inverse(root_quat, world_down),
                quat_apply_inverse(root_quat, root_lin_vel),
                quat_apply_inverse(root_quat, root_ang_vel),
                joint_pos,
                joint_vel,
            ),
            dim=-1,
        )

    def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
        """Extract root world position and xyzw orientation."""
        self._validate_full_state(full_state)
        return full_state[..., :7]

    def apply_prediction(
        self,
        aux_state: torch.Tensor,
        rep_state: torch.Tensor,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate predicted root and joint velocity deltas for one step."""
        self._validate_rep_state(rep_state)
        if prediction.shape[-1] != self.prediction_dim:
            raise ValueError(f"prediction must end in {self.prediction_dim} values")
        z, gravity, root_vel, root_ang_vel, joint_pos, joint_vel = rep_state.split(
            (1, 3, 3, 3, self.num_joints, self.num_joints), dim=-1
        )
        del gravity
        delta_root_vel = prediction[..., :3]
        delta_root_ang_vel = prediction[..., 3:6]
        delta_joint_vel = prediction[..., 6:]
        next_root_vel_local = root_vel + delta_root_vel
        next_root_ang_vel_local = root_ang_vel + delta_root_ang_vel
        next_joint_vel = joint_vel + delta_joint_vel
        next_joint_pos = joint_pos + (joint_vel + self.substep_coeff * delta_joint_vel) * self.dt

        root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
        next_root_vel_world = quat_apply(root_quat, next_root_vel_local)
        next_root_ang_vel_world = quat_apply(root_quat, next_root_ang_vel_local)
        pose_vel_world = quat_apply(root_quat, root_vel + self.substep_coeff * delta_root_vel)
        pose_ang_vel_world = quat_apply(root_quat, root_ang_vel + self.substep_coeff * delta_root_ang_vel)
        next_root_pos = root_pos + pose_vel_world * self.dt
        next_root_quat = quat_integrate(root_quat, pose_ang_vel_world, self.dt)
        next_root_vel = quat_apply_inverse(next_root_quat, next_root_vel_world)
        next_root_ang_vel = quat_apply_inverse(next_root_quat, next_root_ang_vel_world)

        world_down = torch.zeros_like(root_vel)
        world_down[..., 2] = -1.0
        next_gravity = quat_apply_inverse(next_root_quat, world_down)
        next_z = next_root_pos[..., 2:3]
        next_rep_state = torch.cat(
            (
                next_z,
                next_gravity,
                next_root_vel,
                next_root_ang_vel,
                next_joint_pos,
                next_joint_vel,
            ),
            dim=-1,
        )
        next_aux_state = torch.cat((next_root_pos, next_root_quat), dim=-1)
        return next_aux_state, next_rep_state

    def reconstruct_full_state(self, aux_state: torch.Tensor, rep_state: torch.Tensor) -> torch.Tensor:
        """Reconstruct xyzw full state from auxiliary and represented state."""
        self._validate_rep_state(rep_state)
        z, _, root_vel, root_ang_vel, joint_pos, joint_vel = rep_state.split(
            (1, 3, 3, 3, self.num_joints, self.num_joints), dim=-1
        )
        root_quat = aux_state[..., 3:7]
        root_pos = torch.cat((aux_state[..., :2], z), dim=-1)
        velocity = torch.cat(
            (
                quat_apply(root_quat, root_vel),
                quat_apply(root_quat, root_ang_vel),
            ),
            dim=-1,
        )
        return torch.cat((root_pos, root_quat, joint_pos, velocity, joint_vel), dim=-1)

    def loss_blocks(self, rep_state: torch.Tensor, aux_state: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Return named state channels used by the supervised world-model loss."""
        _, _, _, _, joint_pos, joint_vel = rep_state.split((1, 3, 3, 3, self.num_joints, self.num_joints), dim=-1)
        return {
            "root_pos": aux_state[..., :3],
            "root_ori": quat_to_matrix(aux_state[..., 3:7]).flatten(-2) * (0.5**0.5),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }

    def world_frame_bodies(
        self,
        aux_state: torch.Tensor,
        rep_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate differentiable world-frame link poses and velocities."""
        if self.forward_kinematics is None:
            raise RuntimeError("forward kinematics must be bound before model rollouts")
        _, _, root_vel, root_ang_vel, joint_pos, joint_vel = rep_state.split(
            (1, 3, 3, 3, self.num_joints, self.num_joints), dim=-1
        )
        root_quat = aux_state[..., 3:7]
        pos, quat, lin_vel, ang_vel = self.forward_kinematics(
            aux_state[..., :3],
            root_quat,
            joint_pos,
            quat_apply(root_quat, root_vel),
            quat_apply(root_quat, root_ang_vel),
            joint_vel,
        )
        return pos, quat_to_matrix(quat), lin_vel, ang_vel

    def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
        """Return joint positions followed by joint velocities."""
        self._validate_rep_state(rep_state)
        return rep_state[..., 10:]

    def compute_wm_obs(
        self,
        rep_state: torch.Tensor,
        aux_state: torch.Tensor,
        joint_state: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Synthesize projected gravity, root angular velocity, and joint proprio."""
        self._validate_rep_state(rep_state)
        gravity, root_ang_vel = rep_state[..., 1:4], rep_state[..., 7:10]
        joint_pos, joint_vel = self._joint_measurements(joint_state, sim_ctx)
        return torch.cat((gravity, root_ang_vel, joint_pos, joint_vel), dim=-1)

    @staticmethod
    def _validate_full_state_width(full_state: torch.Tensor, expected_dim: int) -> None:
        if full_state.shape[-1] != expected_dim:
            raise ValueError(f"full_state must end in {expected_dim} values")

    def _validate_full_state(self, full_state: torch.Tensor) -> None:
        self._validate_full_state_width(full_state, self.full_state_dim)


class MaxCoordRepresentation(StateRepresentation):
    """Maximal-coordinate link state and root-frame transition model.

    Full state stores all body positions, quaternions, linear velocities and
    angular velocities in separate flattened blocks. The representation stores
    root-relative body states followed by root height and projected up direction.
    """

    LOSS_WEIGHT_CONFIG = {
        **StateRepresentation.LOSS_WEIGHT_CONFIG,
        "body_pos": "nominal_body_position_error",
        "body_rot": "nominal_body_orientation_error",
        "body_vel": "nominal_body_linear_velocity_error",
        "body_avel": "nominal_body_angular_velocity_error",
    }

    def __init__(
        self,
        num_bodies: int,
        default_joint_pos: torch.Tensor,
        dt: float,
        decimation: int,
        analytic_ik: Any | None = None,
    ) -> None:
        super().__init__(default_joint_pos.numel(), default_joint_pos, dt, decimation)
        if num_bodies <= 0:
            raise ValueError("num_bodies must be positive")
        self.num_bodies = num_bodies
        self.full_state_dim = num_bodies * 13
        self.rep_state_dim = num_bodies * 15 + 4
        self.prediction_dim = num_bodies * 6
        self.analytic_ik = analytic_ik

    def bind_analytic_ik(self, analytic_ik: Any) -> None:
        """Attach a robot-specific differentiable analytic IK solver."""
        if not callable(analytic_ik):
            raise TypeError("analytic_ik must be callable")
        self.bind(analytic_ik=analytic_ik)

    def represent(self, full_state: torch.Tensor) -> torch.Tensor:
        """Convert flattened world-frame body states to the root-relative state."""
        self._validate_full_state(full_state)
        num_bodies = self.num_bodies
        positions = full_state[..., : num_bodies * 3].unflatten(-1, (num_bodies, 3))
        quaternions = full_state[..., num_bodies * 3 : num_bodies * 7].unflatten(-1, (num_bodies, 4))
        linear_velocities = full_state[..., num_bodies * 7 : num_bodies * 10].unflatten(-1, (num_bodies, 3))
        angular_velocities = full_state[..., num_bodies * 10 :].unflatten(-1, (num_bodies, 3))

        root_pos = positions[..., :1, :]
        root_quat = quaternions[..., :1, :]
        expanded_root_quat = root_quat.expand_as(quaternions)
        local_pos = quat_apply_inverse(expanded_root_quat, positions - root_pos)
        local_quat = quat_mul(quat_inv(expanded_root_quat), quaternions)
        local_rot6d = quat_to_rot6d(local_quat)
        local_lin_vel = quat_apply_inverse(expanded_root_quat, linear_velocities)
        local_ang_vel = quat_apply_inverse(expanded_root_quat, angular_velocities)
        world_down = torch.zeros_like(positions[..., 0, :])
        world_down[..., 2] = -1.0
        up_dir = quat_apply_inverse(root_quat[..., 0, :], world_down)
        root_height = positions[..., 0, 2:3]
        return torch.cat(
            (
                local_pos.flatten(-2, -1),
                local_rot6d.flatten(-2, -1),
                local_lin_vel.flatten(-2, -1),
                local_ang_vel.flatten(-2, -1),
                root_height,
                up_dir,
            ),
            dim=-1,
        )

    def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
        """Extract root world position and xyzw orientation."""
        self._validate_full_state(full_state)
        root_quat_start = self.num_bodies * 3
        return torch.cat((full_state[..., :3], full_state[..., root_quat_start : root_quat_start + 4]), dim=-1)

    def apply_prediction(
        self,
        aux_state: torch.Tensor,
        rep_state: torch.Tensor,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate predicted link velocity deltas and change to the new root frame."""
        self._validate_rep_state(rep_state)
        if prediction.shape[-1] != self.prediction_dim:
            raise ValueError(f"prediction must end in {self.prediction_dim} values")
        num_bodies = self.num_bodies
        local_pos, local_rot6d, local_lin_vel, local_ang_vel, _, _ = self._decompose_rep_state(rep_state)
        velocity_delta = prediction[..., : num_bodies * 3].unflatten(-1, (num_bodies, 3))
        angular_delta = prediction[..., num_bodies * 3 :].unflatten(-1, (num_bodies, 3))
        pose_lin_vel = local_lin_vel + self.substep_coeff * velocity_delta
        pose_ang_vel = local_ang_vel + self.substep_coeff * angular_delta
        next_lin_vel = local_lin_vel + velocity_delta
        next_ang_vel = local_ang_vel + angular_delta

        root_pos_local = local_pos[..., :1, :]
        relative_velocity = pose_lin_vel - pose_lin_vel[..., :1, :]
        next_pos_local = local_pos - root_pos_local + relative_velocity * self.dt
        axis, cosine, sine = rodrigues_precompute(pose_ang_vel * self.dt)
        rot_columns = local_rot6d.unflatten(-1, (3, 2))
        mid_first = rodrigues_apply(rot_columns[..., 0], axis, cosine, sine)
        mid_second = rodrigues_apply(rot_columns[..., 1], axis, cosine, sine)

        root_pos = aux_state[..., :3]
        root_quat = aux_state[..., 3:7]
        next_root_pos = root_pos + quat_apply(
            root_quat,
            root_pos_local.squeeze(-2) + pose_lin_vel[..., 0, :] * self.dt,
        )
        next_root_quat = quat_integrate(
            root_quat,
            quat_apply(root_quat, pose_ang_vel[..., 0, :]),
            self.dt,
        )

        frame_axis, frame_cosine, frame_sine = rodrigues_precompute(-pose_ang_vel[..., :1, :] * self.dt)
        next_local_pos = rodrigues_apply(next_pos_local, frame_axis, frame_cosine, frame_sine)
        next_local_lin_vel = rodrigues_apply(next_lin_vel, frame_axis, frame_cosine, frame_sine)
        next_local_ang_vel = rodrigues_apply(next_ang_vel, frame_axis, frame_cosine, frame_sine)
        next_local_rot6d = torch.stack(
            (
                rodrigues_apply(mid_first, frame_axis, frame_cosine, frame_sine),
                rodrigues_apply(mid_second, frame_axis, frame_cosine, frame_sine),
            ),
            dim=-1,
        ).flatten(-2, -1)
        world_down = torch.zeros_like(next_root_pos)
        world_down[..., 2] = -1.0
        next_up_dir = quat_apply_inverse(next_root_quat, world_down)
        next_aux = torch.cat((next_root_pos, next_root_quat), dim=-1)
        next_rep = torch.cat(
            (
                next_local_pos.flatten(-2, -1),
                next_local_rot6d.flatten(-2, -1),
                next_local_lin_vel.flatten(-2, -1),
                next_local_ang_vel.flatten(-2, -1),
                next_root_pos[..., 2:3],
                next_up_dir,
            ),
            dim=-1,
        )
        return next_aux, next_rep

    def reconstruct_full_state(self, aux_state: torch.Tensor, rep_state: torch.Tensor) -> torch.Tensor:
        """Reconstruct the flattened world-frame body state."""
        body_pos, body_rot, body_lin_vel, body_ang_vel = self.world_frame_bodies(aux_state, rep_state)
        body_quat = matrix_to_quat(body_rot)
        return torch.cat(
            (
                body_pos.flatten(-2, -1),
                body_quat.flatten(-2, -1),
                body_lin_vel.flatten(-2, -1),
                body_ang_vel.flatten(-2, -1),
            ),
            dim=-1,
        )

    def loss_blocks(self, rep_state: torch.Tensor, aux_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return root and non-root link channels for representation-aware losses."""
        local_pos, local_rot6d, local_lin_vel, local_ang_vel, _, _ = self._decompose_rep_state(rep_state)
        return {
            "root_pos": aux_state[..., :3],
            "root_ori": quat_to_matrix(aux_state[..., 3:7]).flatten(-2, -1) * (0.5**0.5),
            "body_pos": local_pos[..., 1:, :].flatten(-2, -1),
            "body_rot": local_rot6d[..., 1:, :].flatten(-2, -1),
            "body_vel": local_lin_vel[..., 1:, :].flatten(-2, -1),
            "body_avel": local_ang_vel[..., 1:, :].flatten(-2, -1),
        }

    def world_frame_bodies(
        self,
        aux_state: torch.Tensor,
        rep_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recover world-frame body positions, rotations and velocities."""
        local_pos, local_rot6d, local_lin_vel, local_ang_vel, _, _ = self._decompose_rep_state(rep_state)
        root_pos = aux_state[..., :3]
        root_mat = quat_to_matrix(aux_state[..., 3:7]).unsqueeze(-3)

        def rotate(vectors: torch.Tensor) -> torch.Tensor:
            return torch.matmul(root_mat, vectors.unsqueeze(-1)).squeeze(-1)

        return (
            root_pos.unsqueeze(-2) + rotate(local_pos),
            torch.matmul(root_mat, rot6d_to_matrix(local_rot6d)),
            rotate(local_lin_vel),
            rotate(local_ang_vel),
        )

    def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
        """Recover joint positions and velocities using a bound analytic IK solver."""
        if self.analytic_ik is None:
            raise RuntimeError("analytic IK must be bound before requesting MaxCoord joint state")
        _, local_rot6d, _, local_ang_vel, _, _ = self._decompose_rep_state(rep_state)
        rotations = rot6d_to_matrix(local_rot6d)
        joint_pos, joint_vel = self.analytic_ik(rotations, local_ang_vel)
        return torch.cat((joint_pos, joint_vel), dim=-1)

    def compute_wm_obs(
        self,
        rep_state: torch.Tensor,
        aux_state: torch.Tensor,
        joint_state: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Synthesize projected gravity, root angular velocity, and IK joints."""
        self._validate_rep_state(rep_state)
        gravity_world = torch.zeros_like(aux_state[..., :3])
        gravity_world[..., 2] = -1.0
        projected_gravity = quat_apply_inverse(aux_state[..., 3:7], gravity_world)
        _, _, _, local_ang_vel, _, _ = self._decompose_rep_state(rep_state)
        joint_pos_rel, joint_vel = self._joint_measurements(joint_state, sim_ctx)
        return torch.cat(
            (
                projected_gravity,
                local_ang_vel[..., 0, :],
                joint_pos_rel,
                joint_vel,
            ),
            dim=-1,
        )

    def _decompose_rep_state(
        self, rep_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_rep_state(rep_state)
        num_bodies = self.num_bodies
        offset = 0
        local_pos = rep_state[..., offset : offset + num_bodies * 3].unflatten(-1, (num_bodies, 3))
        offset += num_bodies * 3
        local_rot6d = rep_state[..., offset : offset + num_bodies * 6].unflatten(-1, (num_bodies, 6))
        offset += num_bodies * 6
        local_lin_vel = rep_state[..., offset : offset + num_bodies * 3].unflatten(-1, (num_bodies, 3))
        offset += num_bodies * 3
        local_ang_vel = rep_state[..., offset : offset + num_bodies * 3].unflatten(-1, (num_bodies, 3))
        offset += num_bodies * 3
        root_height = rep_state[..., offset : offset + 1]
        up_dir = rep_state[..., offset + 1 : offset + 4]
        return local_pos, local_rot6d, local_lin_vel, local_ang_vel, root_height, up_dir

    def _validate_full_state(self, full_state: torch.Tensor) -> None:
        if full_state.shape[-1] != self.full_state_dim:
            raise ValueError(f"full_state must end in {self.full_state_dim} values")
