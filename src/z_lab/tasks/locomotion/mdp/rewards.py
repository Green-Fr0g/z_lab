# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv



def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize mechanical work at the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    applied_torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_vel) * torch.abs(applied_torque), dim=-1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward moving feet through a target clearance height."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_error = torch.square(asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - target_height)
    foot_speed = torch.linalg.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=2)
    return torch.exp(-torch.sum(foot_z_error * torch.tanh(tanh_mult * foot_speed), dim=1) / std)


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward contact phases that follow the requested alternating gait."""
    sensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0
    phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = torch.cat([((phase + value) % 1.0) for value in offset], dim=-1)
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for index in range(len(sensor_cfg.body_ids)):
        reward += (phases[:, index] < threshold) == contact[:, index]
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        reward *= torch.linalg.norm(command, dim=1) > 0.1
    return reward


def anchor_track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of linear velocity commands for one specified body."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_quat_w.torch[:, asset_cfg.body_ids].squeeze(1)
    body_lin_vel_w = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids].squeeze(1)
    vel_yaw = quat_apply_inverse(yaw_quat(body_quat_w), body_lin_vel_w)
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)



def anchor_track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of yaw velocity commands for one specified body."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids, 2].squeeze(1)
    )
    return torch.exp(-ang_vel_error / std**2)


