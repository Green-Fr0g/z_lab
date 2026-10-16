"""AMP reset, online reference, and recovery assistance event terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import EventTermCfg

from .motion_loader import MotionResetManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot", joint_names=".*")


def _reset_amp_expert_history(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset only the motion-dependent AMP expert history for selected environments.

    Isaac Lab's public ``ObservationManager.reset`` resets every observation
    group. Motion clip changes only invalidate ``amp_expert`` history, while
    policy, critic, and ``amp_policy`` observations remain continuous.
    """
    manager = env.observation_manager
    group_name = "amp_expert"
    if group_name not in manager._group_obs_term_names:
        return

    for term_cfg in manager._group_obs_class_term_cfgs.get(group_name, ()):
        term_cfg.func.reset(env_ids=env_ids)
    for term_name in manager._group_obs_term_names[group_name]:
        delay_buffer = manager._group_obs_term_delay_buffer[group_name].get(term_name)
        if delay_buffer is not None:
            delay_buffer.reset(env_ids)
        history_buffer = manager._group_obs_term_history_buffer[group_name].get(term_name)
        if history_buffer is not None:
            history_buffer.reset(batch_ids=env_ids)


def init_motion_loader(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    root_name: str,
    all_body_names: tuple[str, ...],
    recovery_motion_dir: str | None = None,
    recovery_name_patterns: Sequence[str] = ("fallAndGetUp",),
) -> None:
    """Load motion clips at startup; frame transitions are generated online later."""
    del env_ids
    MotionResetManager.get().init(
        motion_dir=motion_dir,
        device=env.device,
        root_name=root_name,
        all_body_names=all_body_names,
        recovery_motion_dir=recovery_motion_dir,
        recovery_name_patterns=recovery_name_patterns,
        num_envs=env.num_envs,
    )


def reset_from_motion_data(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    root_name: str,
    all_body_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    recovery_motion_dir: str | None = None,
    recovery_env_ratio: float = 0.0,
    recovery_name_patterns: Sequence[str] = ("fallAndGetUp",),
    recovery_max_steps: int = 250,
) -> None:
    """Reset the robot and its online reference motion from matching semantic names."""
    MotionResetManager.get().reset(
        env=env,
        env_ids=env_ids,
        motion_dir=motion_dir,
        root_name=root_name,
        all_body_names=all_body_names,
        asset_cfg=asset_cfg,
        recovery_motion_dir=recovery_motion_dir,
        recovery_env_ratio=recovery_env_ratio,
        recovery_name_patterns=recovery_name_patterns,
        recovery_max_steps=recovery_max_steps,
    )


def advance_motion_reference(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Advance references and clear observation history when a clip changes."""
    changed_env_ids = MotionResetManager.get().advance(env_ids)
    if changed_env_ids.numel() > 0:
        _reset_amp_expert_history(env, changed_env_ids)


def update_recovery_status(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    success_height: float,
    success_angle: float,
    success_hold_steps: int,
) -> None:
    """Mark recovery clips successful after a stable upright pose."""
    del env_ids
    changed_env_ids = MotionResetManager.get().update_recovery_status(
        env,
        success_height=success_height,
        success_angle=success_angle,
        success_hold_steps=success_hold_steps,
    )
    if changed_env_ids.numel() > 0:
        _reset_amp_expert_history(env, changed_env_ids)


class RecoveryAssistForceTerm(ManagerTermBase):
    """Apply an annealed, explicitly configured world-frame z force."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._body_ids = self._asset_cfg.body_ids

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        num_envs = len(range(self._env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        body_ids = self._body_ids
        if isinstance(body_ids, slice):
            body_ids = list(range(self._asset.num_bodies))[body_ids]
        forces = torch.zeros((num_envs, len(body_ids), 3), device=self._env.device)
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torch.zeros_like(forces),
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=True,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        force_start: float = 0.0,
        force_end: float = 0.0,
        anneal_steps: int = 50000,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        if len(env_ids) == 0:
            return
        motion = MotionResetManager.get()
        active = (
            motion.recovery_active[env_ids] & ~motion.recovery_succeeded[env_ids] & ~motion.recovery_failed[env_ids]
        )
        body_ids = self._body_ids
        if isinstance(body_ids, slice):
            body_ids = list(range(self._asset.num_bodies))[body_ids]
        body_ids = list(body_ids)
        forces = torch.zeros((len(env_ids), len(body_ids), 3), device=env.device)
        torques = torch.zeros_like(forces)
        force_magnitude = MotionResetManager.get().assist_scale(
            env,
            force_start=force_start,
            force_end=force_end,
            anneal_steps=anneal_steps,
        )
        forces[:, :, 2] = force_magnitude * active[:, None]
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=True,
        )
