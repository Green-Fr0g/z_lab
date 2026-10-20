# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP rewards: command tracking plus recovery-aware auxiliary signals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def recovery_window_mask(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Return which environments are currently being held in a recovery window.

    A recovery window is the period after an environment has been terminated but before its reset
    actually happens. Delayed terminations keep the robot lying on the ground for a few extra steps so
    the policy can learn to get back up.

    Returns ``None`` when the environment does not use delayed resets, so callers can skip scaling.
    """
    manager = getattr(env, "termination_manager", None)
    delay_env_mask = getattr(manager, "delay_env_mask", None)
    delay_counters = getattr(manager, "delay_counters", None)
    if isinstance(delay_env_mask, torch.Tensor) and isinstance(delay_counters, torch.Tensor):
        return delay_env_mask & (delay_counters > 0)
    return None


def track_anchor_linear_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    anchor_cfg: SceneEntityCfg,
    scale_in_recovery: bool = True,
    recovery_reward_scale: float = 0.5,
) -> torch.Tensor:
    """Reward planar velocity tracking measured at the anchor body.

    The commanded z velocity is assumed to be zero, so the command is rotated by the anchor yaw and
    compared against the anchor's world linear velocity.

    Args:
        std: Gaussian width of the tracking error.
        command_name: Name of the velocity command term.
        anchor_cfg: Body used as the measurement point, only the first body is used.
        scale_in_recovery: Multiply the reward of recovering environments by ``recovery_reward_scale``.
            Set to ``False`` to hand out the plain reward everywhere.
        recovery_reward_scale: Reward multiplier for environments held in a recovery window. A robot on
            the ground cannot follow a walking command, so this keeps the tracking term from dominating
            while it is getting back up.
    """
    asset = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    anchor_id = anchor_cfg.body_ids[0]

    anchor_quat_w = asset.data.body_link_quat_w.torch[:, anchor_id]
    anchor_lin_vel_w = asset.data.body_link_lin_vel_w.torch[:, anchor_id]

    # keep the command planar: the z component is always zero
    command_lin_vel_b = torch.cat([command[:, :2], torch.zeros_like(command[:, :1])], dim=-1)
    command_lin_vel_w = quat_apply(yaw_quat(anchor_quat_w), command_lin_vel_b)

    error = torch.sum(torch.square(command_lin_vel_w - anchor_lin_vel_w), dim=-1)
    reward = torch.exp(-error / std**2)

    if scale_in_recovery:
        mask = recovery_window_mask(env)
        if mask is not None:
            reward = torch.where(mask, reward * recovery_reward_scale, reward)
    return reward


def track_anchor_angular_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    anchor_cfg: SceneEntityCfg,
    scale_in_recovery: bool = True,
    recovery_reward_scale: float = 0.5,
) -> torch.Tensor:
    """Reward yaw-rate tracking while penalizing anchor roll and pitch rates.

    Args:
        std: Gaussian width of the combined yaw and roll/pitch error.
        command_name: Name of the velocity command term.
        anchor_cfg: Body used as the measurement point, only the first body is used.
        scale_in_recovery: Multiply the reward of recovering environments by ``recovery_reward_scale``.
        recovery_reward_scale: Reward multiplier for environments held in a recovery window.
    """
    asset = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    anchor_id = anchor_cfg.body_ids[0]

    anchor_quat_w = asset.data.body_link_quat_w.torch[:, anchor_id]
    anchor_ang_vel_w = asset.data.body_link_ang_vel_w.torch[:, anchor_id]

    # yaw error uses world-frame rates, so a level robot tracks the command directly
    ang_vel_z_error = torch.square(command[:, 2] - anchor_ang_vel_w[:, 2])
    # roll and pitch are expressed in the anchor frame to punish tilting
    anchor_ang_vel_b = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)
    ang_vel_xy_error = torch.sum(torch.square(anchor_ang_vel_b[:, :2]), dim=-1)

    reward = torch.exp(-(ang_vel_z_error + ang_vel_xy_error) / std**2)

    if scale_in_recovery:
        mask = recovery_window_mask(env)
        if mask is not None:
            reward = torch.where(mask, reward * recovery_reward_scale, reward)
    return reward


def track_root_height(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
    scale_in_recovery: bool = True,
    recovery_reward_scale: float = 1.0,
) -> torch.Tensor:
    """Reward standing at the robot's default root height.

    This term is the main signal for getting back up, so it is paid only to environments held in a
    recovery window and pays nothing to robots that are already walking normally.

    Args:
        std: Gaussian width of the height error.
        asset_cfg: Robot asset to read the root height from.
        scale_in_recovery: Restrict payment to recovering environments. Set to ``False`` to reward the
            root height of every environment instead.
        recovery_reward_scale: Reward multiplier for environments held in a recovery window.
    """
    asset = env.scene[asset_cfg.name]

    desired_height = asset.data.default_root_state.torch[:, 2]
    current_height = asset.data.root_pos_w.torch[:, 2]

    reward = torch.exp(-torch.square(desired_height - current_height) / std**2)

    if scale_in_recovery:
        mask = recovery_window_mask(env)
        if mask is not None:
            reward = torch.where(mask, reward * recovery_reward_scale, torch.zeros_like(reward))
    return reward


def body_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    std: float,
    body_cfg: SceneEntityCfg,
    scale_in_recovery: bool = True,
    recovery_reward_scale: float = 0.5,
) -> torch.Tensor:
    """Reward keeping the selected bodies level by punishing body-frame roll and pitch rates.

    Args:
        std: Gaussian width of the roll and pitch rate error.
        body_cfg: Bodies whose rotation rates are summed.
        scale_in_recovery: Multiply the reward of recovering environments by ``recovery_reward_scale``.
        recovery_reward_scale: Reward multiplier for environments held in a recovery window.
    """
    asset = env.scene[body_cfg.name]

    body_quat_w = asset.data.body_link_quat_w.torch[:, body_cfg.body_ids]
    body_ang_vel_w = asset.data.body_link_ang_vel_w.torch[:, body_cfg.body_ids]
    body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)

    error = torch.sum(torch.square(body_ang_vel_b[..., :2]), dim=-1).sum(dim=-1)
    reward = torch.exp(-error / std**2)

    if scale_in_recovery:
        mask = recovery_window_mask(env)
        if mask is not None:
            reward = torch.where(mask, reward * recovery_reward_scale, reward)
    return reward
