"""Termination terms for mixed locomotion and recovery AMP episodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .motion_loader import MotionResetManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _recovery_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    motion = MotionResetManager.get()
    return motion.recovery_active


def _value(value):
    return value.torch if hasattr(value, "torch") else value


def recovery_aware_root_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate low roots for normal clips while allowing recovery clips to stand up."""
    asset: Articulation = env.scene[asset_cfg.name]
    height = _value(asset.data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]
    return (height < minimum_height) & ~_recovery_mask(env)


def recovery_aware_bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate tilted normal clips while allowing a recovery clip to remain active."""
    asset: Articulation = env.scene[asset_cfg.name]
    tilt = torch.acos((-_value(asset.data.projected_gravity_b)[:, 2]).clamp(-1.0, 1.0))
    return (tilt > limit_angle) & ~_recovery_mask(env)


def recovery_aware_illegal_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Apply illegal-contact termination only to normal motion clips."""
    sensor = env.scene.sensors[sensor_cfg.name]
    contact = torch.any(
        torch.linalg.norm(_value(sensor.data.net_forces_w)[:, sensor_cfg.body_ids], dim=-1) > threshold,
        dim=-1,
    )
    return contact & ~_recovery_mask(env)


def recovery_failure(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate a recovery episode when its reference clip reaches its final frame."""
    motion = MotionResetManager.get()
    failed = motion.recovery_active & motion.recovery_clip_finished()
    if failed.any():
        motion.mark_failure(failed.nonzero(as_tuple=False).squeeze(-1))
    return failed
