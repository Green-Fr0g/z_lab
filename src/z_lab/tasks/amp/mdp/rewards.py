"""AMP locomotion reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .motion_loader import MotionResetManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_DEFAULT_BODY_CFG = SceneEntityCfg("robot", body_names=())


def _value(value):
    """Return the torch view for Isaac Lab 3.0 proxy arrays."""
    return value.torch if hasattr(value, "torch") else value


def track_anchor_linear_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    anchor_cfg: SceneEntityCfg = _DEFAULT_BODY_CFG,
) -> torch.Tensor:
    """Reward planar velocity tracking at the configured anchor body."""
    asset: Articulation = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    anchor_quat_w = _value(asset.data.body_link_quat_w)[:, anchor_cfg.body_ids[0]]
    anchor_lin_vel_w = _value(asset.data.body_link_lin_vel_w)[:, anchor_cfg.body_ids[0], :3]
    command_lin_vel_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
    command_lin_vel_w = quat_apply(yaw_quat(anchor_quat_w), command_lin_vel_b)
    return torch.exp(-torch.sum(torch.square(command_lin_vel_w - anchor_lin_vel_w), dim=-1) / std**2)


def track_anchor_angular_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    anchor_cfg: SceneEntityCfg = _DEFAULT_BODY_CFG,
) -> torch.Tensor:
    """Reward yaw-rate tracking while suppressing anchor roll and pitch rates."""
    asset: Articulation = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    anchor_quat_w = _value(asset.data.body_link_quat_w)[:, anchor_cfg.body_ids[0]]
    anchor_ang_vel_w = _value(asset.data.body_link_ang_vel_w)[:, anchor_cfg.body_ids[0]]
    ang_vel_z_error = torch.square(command[:, 2] - anchor_ang_vel_w[:, 2])
    anchor_ang_vel_b = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)
    ang_vel_xy_error = torch.sum(torch.square(anchor_ang_vel_b[:, :2]), dim=-1)
    return torch.exp(-(ang_vel_z_error + ang_vel_xy_error) / std**2)


def body_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg = _DEFAULT_BODY_CFG,
) -> torch.Tensor:
    """Penalize body-frame angular velocity around the x and y axes."""
    asset: Articulation = env.scene[body_cfg.name]
    body_quat_w = _value(asset.data.body_link_quat_w)[:, body_cfg.body_ids]
    body_ang_vel_w = _value(asset.data.body_link_ang_vel_w)[:, body_cfg.body_ids]
    body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)
    return torch.sum(torch.square(body_ang_vel_b[..., :2]), dim=-1).sum(dim=-1)


class recovery_success_rate(ManagerTermBase):
    """Log recovery success without contributing to the task reward."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def reset(self, env_ids: torch.Tensor) -> None:
        motion = MotionResetManager.get()
        del env_ids
        success_rate, active_ratio = motion.last_recovery_metrics
        log = self._env.extras.setdefault("log", {})
        log["Metrics/recovery_success_rate"] = success_rate
        log["Metrics/recovery_active_ratio"] = active_ratio

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return torch.zeros(env.num_envs, device=env.device)
