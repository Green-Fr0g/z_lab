# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Connect motion loading, robot resets, and delayed terminations to environment events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .motion_loader import MotionLoader
from .terminations import DelayedTerminationManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["init_amp_motion", "init_delayed_termination", "reset_from_amp_motion"]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def init_amp_motion(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    motion_dir: str,
    recovery_dir: str | None = None,
    recovery_name_patterns: Sequence[str] = ("fall", "getup", "get_up", "recovery"),
    root_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Load raw clips at startup and publish ``env.amp_motion_loader``.

    All bodies and joints are aligned to the selected articulation's full simulation order.
    ``env_ids`` is unused: the loaded pools are shared by all environments on ``env.device``.
    Expert feature transforms and training datasets are constructed by the training layer.
    """
    asset = env.scene[asset_cfg.name]
    env.amp_motion_loader = MotionLoader(
        motion_dir=motion_dir,
        body_names=asset.body_names,
        joint_names=asset.joint_names,
        device=env.device,
        root_name=root_name,
        recovery_dir=recovery_dir,
        recovery_name_patterns=recovery_name_patterns,
    )


def init_delayed_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    delay_reset_env_ratio: float = 0.0,
    max_delay_steps: int = 0,
) -> None:
    """Install a delay manager at startup for a fixed random subset of environments.

    ``delay_reset_env_ratio`` is in ``[0, 1]``. A zero ratio or zero step count disables the
    delay. The environment's timeout signal always takes priority over the recovery window.
    ``env_ids`` is unused because the mask is selected once for the entire run.
    """
    if not 0.0 <= delay_reset_env_ratio <= 1.0:
        raise ValueError("delay_reset_env_ratio must be in [0, 1].")
    if isinstance(max_delay_steps, bool) or not isinstance(max_delay_steps, int) or max_delay_steps < 0:
        raise ValueError("max_delay_steps must be a nonnegative integer.")
    num_delay = int(env.num_envs * delay_reset_env_ratio)
    if num_delay == 0 or max_delay_steps == 0:
        return
    delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    delay_mask[torch.randperm(env.num_envs, device=env.device)[:num_delay]] = True
    env.termination_manager = DelayedTerminationManager(env.termination_manager, delay_mask, max_delay_steps)


@torch.no_grad()
def reset_from_amp_motion(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    use_recovery: bool = True,
) -> None:
    """Write a sampled root and joint state to the requested environments.

    Delayed environments use recovery clips when ``use_recovery`` is true; all other environments
    use normal clips. Missing recovery clips fall back to normal motion. Root positions receive
    the environment-origin offset; joint positions are clamped to the simulation's soft limits.
    Both loading and reset must select the same articulation. All its joints are reset in simulation
    order, regardless of ``asset_cfg.joint_ids``.
    """
    loader = getattr(env, "amp_motion_loader", None)
    if loader is None:
        raise RuntimeError("reset_from_amp_motion requires init_amp_motion as a startup event.")
    if env_ids is None or isinstance(env_ids, slice):
        selector = slice(None) if env_ids is None else env_ids
        env_ids = torch.arange(env.num_envs, device=env.device)[selector]
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if env_ids.numel() == 0:
        return

    manager = env.termination_manager
    recovery_mask = (
        manager.delay_env_mask[env_ids] if use_recovery and isinstance(manager, DelayedTerminationManager) else None
    )
    state = loader.sample_reset_state(len(env_ids), recovery_mask)
    asset = env.scene[asset_cfg.name]
    root_pose = torch.cat([state["root_pos"] + env.scene.env_origins[env_ids], state["root_quat"]], dim=-1)
    root_velocity = torch.cat([state["root_lin_vel"], state["root_ang_vel"]], dim=-1)
    limits = asset.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = state["dof_pos"].clamp(limits[..., 0], limits[..., 1])

    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, state["dof_vel"], env_ids=env_ids)
