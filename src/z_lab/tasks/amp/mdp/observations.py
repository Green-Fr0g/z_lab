# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Body-frame AMP state features shared by the policy observations and the discriminator.

All four terms flatten the selected bodies in the order requested by the scene entity
configuration, so the environment and the expert motion loader produce the same feature layout as
long as they select the same body names. Quaternions follow Isaac Lab's ``xyzw`` convention and
positions use world coordinates before the anchor transform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "robot_body_pos_b",
    "robot_body_ori_b",
    "robot_body_lin_vel_b",
    "robot_body_ang_vel_b",
]

_DEFAULT_ANCHOR_CFG = SceneEntityCfg("robot", body_names=())
_DEFAULT_BODY_CFG = SceneEntityCfg("robot", body_names=())


def _anchor_frame(
    env: ManagerBasedEnv, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return anchor position/orientation and the selected body poses in world frame."""
    asset = env.scene[anchor_cfg.name]
    anchor_pos_w = asset.data.body_link_pos_w.torch[:, anchor_cfg.body_ids[0]]
    anchor_quat_w = asset.data.body_link_quat_w.torch[:, anchor_cfg.body_ids[0]]
    body_pos_w = asset.data.body_link_pos_w.torch[:, body_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w.torch[:, body_cfg.body_ids]
    return anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w


def robot_body_pos_b(
    env: ManagerBasedEnv,
    anchor_cfg: SceneEntityCfg = _DEFAULT_ANCHOR_CFG,
    body_cfg: SceneEntityCfg = _DEFAULT_BODY_CFG,
) -> torch.Tensor:
    """Body positions expressed in the anchor frame, flattened per body."""
    anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w = _anchor_frame(env, anchor_cfg, body_cfg)
    num_bodies = body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ManagerBasedEnv,
    anchor_cfg: SceneEntityCfg = _DEFAULT_ANCHOR_CFG,
    body_cfg: SceneEntityCfg = _DEFAULT_BODY_CFG,
) -> torch.Tensor:
    """Body orientations relative to the anchor, stored as the first two rotation columns."""
    anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w = _anchor_frame(env, anchor_cfg, body_cfg)
    num_bodies = body_pos_w.shape[1]
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    matrix = matrix_from_quat(ori_b)
    return matrix[..., :2].reshape(env.num_envs, num_bodies * 6)


def robot_body_lin_vel_b(
    env: ManagerBasedEnv,
    anchor_cfg: SceneEntityCfg = _DEFAULT_ANCHOR_CFG,
    body_cfg: SceneEntityCfg = _DEFAULT_BODY_CFG,
) -> torch.Tensor:
    """Body linear velocities expressed in each body's own frame, flattened per body."""
    del anchor_cfg
    asset = env.scene[body_cfg.name]
    body_lin_vel_w = asset.data.body_link_lin_vel_w.torch[:, body_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w.torch[:, body_cfg.body_ids]
    num_bodies = body_lin_vel_w.shape[1]
    body_lin_vel_b = quat_apply_inverse(body_quat_w.reshape(-1, 4), body_lin_vel_w.reshape(-1, 3)).reshape(
        env.num_envs, num_bodies, 3
    )
    return body_lin_vel_b.reshape(env.num_envs, num_bodies * 3)


def robot_body_ang_vel_b(
    env: ManagerBasedEnv,
    anchor_cfg: SceneEntityCfg = _DEFAULT_ANCHOR_CFG,
    body_cfg: SceneEntityCfg = _DEFAULT_BODY_CFG,
) -> torch.Tensor:
    """Body angular velocities expressed in each body's own frame, flattened per body."""
    del anchor_cfg
    asset = env.scene[body_cfg.name]
    body_ang_vel_w = asset.data.body_link_ang_vel_w.torch[:, body_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w.torch[:, body_cfg.body_ids]
    num_bodies = body_ang_vel_w.shape[1]
    body_ang_vel_b = quat_apply_inverse(body_quat_w.reshape(-1, 4), body_ang_vel_w.reshape(-1, 3)).reshape(
        env.num_envs, num_bodies, 3
    )
    return body_ang_vel_b.reshape(env.num_envs, num_bodies * 3)
