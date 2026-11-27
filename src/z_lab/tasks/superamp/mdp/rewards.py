"""Velocity tracking terms for SuperAMP locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def track_anchor_lin_vel_xy(
    env: ManagerBasedRLEnv, command_name: str, anchor_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """Reward planar torso velocity tracking in the root link's yaw frame."""
    data = env.scene[anchor_cfg.name].data
    velocity = data.body_link_lin_vel_w.torch[:, anchor_cfg.body_ids[0]]
    velocity_b = quat_apply_inverse(yaw_quat(data.root_link_quat_w.torch), velocity)
    target = env.command_manager.get_command(command_name)[:, :2]
    return torch.exp(-(velocity_b[:, :2] - target).square().sum(dim=-1) / std**2)
