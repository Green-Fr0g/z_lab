# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP reward terms measured at a configurable anchor body.

Velocity tracking uses the anchor link rather than the root so that the command error matches the
motion features the discriminator sees. The auxiliary angular-velocity penalty keeps the selected
bodies upright without constraining yaw.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "track_anchor_linear_velocity",
    "track_anchor_angular_velocity",
    "body_ang_vel_xy_l2",
]

_DEFAULT_ANCHOR_CFG = SceneEntityCfg("robot", body_names=())
_DEFAULT_BODY_CFG = SceneEntityCfg("robot", body_names=())


def track_anchor_linear_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    anchor_cfg: SceneEntityCfg = _DEFAULT_ANCHOR_CFG,
) -> torch.Tensor:
    """Reward tracking commanded planar linear velocity at an anchor body."""
    asset: Articulation = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    anchor_quat_w = asset.data.body_link_quat_w.torch[:, anchor_cfg.body_ids[0]]
    anchor_lin_vel_w = asset.data.body_link_lin_vel_w.torch[:, anchor_cfg.body_ids[0], :3]
    command_lin_vel_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
    command_lin_vel_w = quat_apply(yaw_quat(anchor_quat_w), command_lin_vel_b)
    lin_vel_error = torch.sum(torch.square(command_lin_vel_w - anchor_lin_vel_w), dim=-1)
    return torch.exp(-lin_vel_error / std**2)


def track_anchor_angular_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    anchor_cfg: SceneEntityCfg = _DEFAULT_ANCHOR_CFG,
) -> torch.Tensor:
    """Reward tracking commanded yaw rate while suppressing roll and pitch rates."""
    asset: Articulation = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    anchor_quat_w = asset.data.body_link_quat_w.torch[:, anchor_cfg.body_ids[0]]
    anchor_ang_vel_w = asset.data.body_link_ang_vel_w.torch[:, anchor_cfg.body_ids[0]]
    ang_vel_z_error = torch.square(command[:, 2] - anchor_ang_vel_w[:, 2])
    anchor_ang_vel_b = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)
    ang_vel_xy_error = torch.sum(torch.square(anchor_ang_vel_b[:, :2]), dim=-1)
    return torch.exp(-(ang_vel_z_error + ang_vel_xy_error) / std**2)


def body_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    std: float,
    body_cfg: SceneEntityCfg = _DEFAULT_BODY_CFG,
) -> torch.Tensor:
    """Reward low body-frame x/y angular velocity with an exponential term.

    This matches the AMP_mjlab implementation: the selected body's angular
    velocity is expressed in its body frame, the x/y squared error is scaled
    by ``std``, and the result is returned as a positive exponential reward.
    """
    asset: Articulation = env.scene[body_cfg.name]
    body_id = body_cfg.body_ids[0]
    body_quat_w = asset.data.body_link_quat_w.torch[:, body_id]
    body_ang_vel_w = asset.data.body_link_ang_vel_w.torch[:, body_id]
    body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)
    ang_vel_xy_error = torch.sum(torch.square(body_ang_vel_b[:, :2]), dim=-1)
    return torch.exp(-ang_vel_xy_error / std**2)
