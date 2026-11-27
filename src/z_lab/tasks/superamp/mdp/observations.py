"""World-model state observations for velocity-conditioned locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def gencoord_state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the root link pose, joint positions, world-frame root twist, and joint velocities."""
    data = env.scene[asset_cfg.name].data
    joint_ids = asset_cfg.joint_ids
    return torch.cat(
        (
            data.root_link_pos_w.torch - env.scene.env_origins,
            data.root_link_quat_w.torch,
            data.joint_pos.torch[:, joint_ids],
            data.root_link_lin_vel_w.torch,
            data.root_link_ang_vel_w.torch,
            data.joint_vel.torch[:, joint_ids],
        ),
        dim=-1,
    )


def maxcoord_state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return blocks of link positions, quaternions, and world-frame twists in configured body order."""
    data = env.scene[asset_cfg.name].data
    body_ids = asset_cfg.body_ids
    positions = data.body_link_pos_w.torch[:, body_ids] - env.scene.env_origins[:, None]
    orientations = data.body_link_quat_w.torch[:, body_ids]
    linear_velocities = data.body_link_lin_vel_w.torch[:, body_ids]
    angular_velocities = data.body_link_ang_vel_w.torch[:, body_ids]
    return torch.cat(
        (positions.flatten(1), orientations.flatten(1), linear_velocities.flatten(1), angular_velocities.flatten(1)),
        dim=-1,
    )
