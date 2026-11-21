"""Differentiable motion-tracking reward used by the world-model rollout."""

from __future__ import annotations

import torch

from supermimic.utils.math import (
    quat_apply,
    quat_inv,
    quat_mul,
    quat_to_matrix,
    quat_yaw,
    rotation_error_magnitude,
)


def _reanchor_targets(
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    predicted_root_pos: torch.Tensor,
    predicted_root_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move body targets with predicted root XY and yaw, retaining target tilt."""
    reference_root_pos = target_pos[..., 0, :]
    reference_root_quat = target_quat[..., 0, :]
    delta_yaw = quat_yaw(quat_mul(predicted_root_quat, quat_inv(reference_root_quat)))
    delta_pos = torch.cat((predicted_root_pos[..., :2], reference_root_pos[..., 2:3]), dim=-1)
    relative_pos = target_pos - reference_root_pos.unsqueeze(-2)
    anchored_pos = delta_pos.unsqueeze(-2) + quat_apply(delta_yaw.unsqueeze(-2), relative_pos)
    anchored_quat = quat_mul(delta_yaw.unsqueeze(-2), target_quat)
    return anchored_pos, quat_to_matrix(anchored_quat)


def tracking_reward(
    aux_state: torch.Tensor,
    predicted_bodies: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    action: torch.Tensor,
    last_action: torch.Tensor,
    joint_positions: torch.Tensor,
    target_body_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    soft_joint_pos_limits: torch.Tensor,
    step_dt: float,
    weights: dict[str, float],
    stds: dict[str, float],
) -> torch.Tensor:
    """Compute the motion-tracking and regularization reward for one step."""
    pred_pos, pred_rot, pred_lin_vel, pred_ang_vel = predicted_bodies
    target_pos, target_quat, target_lin_vel, target_ang_vel = target_body_state
    target_rot = quat_to_matrix(target_quat)
    anchored_pos, anchored_rot = _reanchor_targets(target_pos, target_quat, aux_state[..., :3], aux_state[..., 3:7])

    errors = {
        "motion_global_anchor_pos": (pred_pos[..., 0, :] - target_pos[..., 0, :]).square().sum(-1),
        "motion_global_anchor_ori": rotation_error_magnitude(pred_rot[..., 0, :, :], target_rot[..., 0, :, :]).square(),
        "motion_body_pos": (pred_pos - anchored_pos).square().sum(-1).mean(-1),
        "motion_body_ori": rotation_error_magnitude(pred_rot, anchored_rot).square().mean(-1),
        "motion_body_lin_vel": (pred_lin_vel - target_lin_vel).square().sum(-1).mean(-1),
        "motion_body_ang_vel": (pred_ang_vel - target_ang_vel).square().sum(-1).mean(-1),
    }
    reward = sum(weights[name] * torch.exp(-error / stds[name] ** 2) for name, error in errors.items())
    lower, upper = soft_joint_pos_limits.unbind(-1)
    limit_error = ((lower - joint_positions).clamp_min(0) + (joint_positions - upper).clamp_min(0)).sum(-1)
    action_rate = (action - last_action).square().sum(-1)
    return step_dt * (reward + weights["action_rate_l2"] * action_rate + weights["joint_limit"] * limit_error)
