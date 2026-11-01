"""AMP locomotion rewards adapted from AMP_mjlab."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _delayed_failure_mask(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Select recovery-enabled environments currently waiting for failure termination."""
    manager = env.termination_manager
    delay_mask = getattr(manager, "_delay_env_mask", None)
    counters = getattr(manager, "_delay_counters", None)
    if delay_mask is None or counters is None:
        return None
    return delay_mask & (counters > 0)


def _scale_delayed_reward(
    env: ManagerBasedRLEnv, reward: torch.Tensor, mask_delay: bool, delay_env_rew_ratio: float
) -> torch.Tensor:
    """Scale rewards only while an environment is in delayed failure termination."""
    mask = _delayed_failure_mask(env) if mask_delay else None
    if mask is None:
        return reward
    return torch.where(mask, reward * delay_env_rew_ratio, reward)


def track_anchor_linear_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    anchor_cfg: SceneEntityCfg,
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 1.0,
) -> torch.Tensor:
    """Track anchor velocity, optionally disabling tracking during recovery attempts."""
    asset = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    anchor_quat = asset.data.body_link_quat_w.torch[:, anchor_cfg.body_ids[0]]
    actual = asset.data.body_link_lin_vel_w.torch[:, anchor_cfg.body_ids[0], :3]
    target_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
    target_w = quat_apply(yaw_quat(anchor_quat), target_b)
    error = torch.sum(torch.square(target_w - actual), dim=-1)
    reward = torch.exp(-error / std**2)
    return _scale_delayed_reward(env, reward, mask_delay, delay_env_rew_ratio)


def track_anchor_angular_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    anchor_cfg: SceneEntityCfg,
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 1.0,
) -> torch.Tensor:
    """Track yaw velocity and suppress anchor roll/pitch angular velocity."""
    asset = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    quat = asset.data.body_link_quat_w.torch[:, anchor_cfg.body_ids[0]]
    vel_w = asset.data.body_link_ang_vel_w.torch[:, anchor_cfg.body_ids[0]]
    err_z = torch.square(command[:, 2] - vel_w[:, 2])
    vel_b = quat_apply_inverse(quat, vel_w)
    reward = torch.exp(-(err_z + torch.sum(torch.square(vel_b[:, :2]), dim=-1)) / std**2)
    return _scale_delayed_reward(env, reward, mask_delay, delay_env_rew_ratio)


def body_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    std: float,
    body_cfg: SceneEntityCfg,
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 1.0,
) -> torch.Tensor:
    """Reward low body roll/pitch angular velocity outside delayed recovery attempts."""
    asset = env.scene[body_cfg.name]
    quat = asset.data.body_link_quat_w.torch[:, body_cfg.body_ids[0]]
    vel_w = asset.data.body_link_ang_vel_w.torch[:, body_cfg.body_ids[0]]
    vel_b = quat_apply_inverse(quat, vel_w)
    reward = torch.exp(-torch.sum(torch.square(vel_b[:, :2]), dim=-1) / std**2)
    return _scale_delayed_reward(env, reward, mask_delay, delay_env_rew_ratio)


def track_root_height(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 1.0,
) -> torch.Tensor:
    """Reward standing height only during delayed recovery, as in AMP_mjlab."""
    asset = env.scene[asset_cfg.name]
    target = asset.data.default_root_state[:, 2]
    height = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids[0], 2]
    reward = torch.exp(-torch.square(target - height) / std**2)
    mask = _delayed_failure_mask(env) if mask_delay else None
    if mask is None:
        return torch.zeros_like(reward)
    return torch.where(mask, reward * delay_env_rew_ratio, torch.zeros_like(reward))
