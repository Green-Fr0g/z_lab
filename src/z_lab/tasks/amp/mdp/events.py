"""AMP expert motion loading and motion-based environment resets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg, TerminationManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionResetManager:
    """Load G1_23 expert clips once and reset from name-aligned recorded frames."""

    _clips: dict[str, list[dict[str, torch.Tensor | tuple[str, ...]]]] = {}
    _recovery_clips: dict[str, list[dict[str, torch.Tensor | tuple[str, ...]]]] = {}

    @classmethod
    def init(
        cls,
        motion_dir: str,
        device: str,
        recovery_motion_names: tuple[str, ...] = (),
    ):
        key = str(Path(motion_dir).resolve())
        if key in cls._clips:
            return key
        paths = sorted(Path(key).rglob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No AMP NPZ files found under {key}")
        clips = []
        recovery_clips = []
        required = (
            "body_names",
            "dof_names",
            "body_positions",
            "body_rotations",
            "body_linear_velocities",
            "body_angular_velocities",
            "dof_positions",
            "dof_velocities",
        )
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                missing = [name for name in required if name not in data]
                if missing:
                    raise KeyError(f"{path} is missing AMP fields: {missing}")
                recorded_bodies = tuple(str(name) for name in data["body_names"].tolist())
                clip = {
                    "path": str(path),
                    "body_names": recorded_bodies,
                    "dof_names": tuple(str(name) for name in data["dof_names"].tolist()),
                    "body_positions": torch.as_tensor(data["body_positions"], dtype=torch.float32, device=device),
                    "body_rotations": torch.as_tensor(data["body_rotations"], dtype=torch.float32, device=device),
                    "body_linear_velocities": torch.as_tensor(
                        data["body_linear_velocities"], dtype=torch.float32, device=device
                    ),
                    "body_angular_velocities": torch.as_tensor(
                        data["body_angular_velocities"], dtype=torch.float32, device=device
                    ),
                    "dof_positions": torch.as_tensor(data["dof_positions"], dtype=torch.float32, device=device),
                    "dof_velocities": torch.as_tensor(data["dof_velocities"], dtype=torch.float32, device=device),
                }
                (recovery_clips if path.stem in recovery_motion_names else clips).append(clip)
        if not clips:
            raise ValueError(f"No normal AMP motion files found under {key}")
        cls._clips[key] = clips
        cls._recovery_clips[key] = recovery_clips
        return key

    @classmethod
    def reset(
        cls,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice | None,
        motion_dir: str,
        root_name: str,
        asset_cfg: SceneEntityCfg,
        recovery_motion_names: tuple[str, ...] = (),
    ) -> None:
        key = str(Path(motion_dir).resolve())
        if key not in cls._clips:
            cls.init(motion_dir, str(env.device), recovery_motion_names)
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        asset: Articulation = env.scene[asset_cfg.name]
        joint_names = tuple(asset.joint_names)
        joint_pos = torch.empty((len(env_ids), len(joint_names)), device=env.device)
        joint_vel = torch.empty_like(joint_pos)
        root_pos = torch.empty((len(env_ids), 3), device=env.device)
        root_quat = torch.empty((len(env_ids), 4), device=env.device)
        root_lin_vel = torch.empty_like(root_pos)
        root_ang_vel = torch.empty_like(root_pos)
        delay_mask = getattr(env.termination_manager, "_delay_env_mask", None)
        is_delay = delay_mask[env_ids] if delay_mask is not None else torch.zeros_like(env_ids, dtype=torch.bool)
        if not hasattr(env, "amp_recovery_env_mask"):
            env.amp_recovery_env_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # A delayed-termination environment may fall back to a normal clip if recovery data is absent.
        env.amp_recovery_env_mask[env_ids] = is_delay & bool(cls._recovery_clips[key])
        for select, clips in ((~is_delay, cls._clips[key]), (is_delay, cls._recovery_clips[key])):
            selected_ids = env_ids[select]
            if selected_ids.numel() == 0:
                continue
            if not clips:
                clips = cls._clips[key]
            clip_lengths = torch.tensor(
                [clip["dof_positions"].shape[0] for clip in clips], device=env.device, dtype=torch.float
            )
            clip_ids = torch.multinomial(clip_lengths / clip_lengths.sum(), len(selected_ids), replacement=True)
            frame_ids = torch.empty_like(clip_ids)
            for clip_id in clip_ids.unique().tolist():
                mask = clip_ids == clip_id
                frame_ids[mask] = torch.randint(
                    clips[clip_id]["dof_positions"].shape[0], (int(mask.sum()),), device=env.device
                )
            for clip_id in clip_ids.unique().tolist():
                mask = clip_ids == clip_id
                clip = clips[clip_id]
                frames = frame_ids[mask]
                body_idx = clip["body_names"].index(root_name)
                clip_joint_indexes = [clip["dof_names"].index(name) for name in joint_names]
                target_rows = torch.nonzero(select, as_tuple=False).squeeze(-1)[mask]
                root_pos[target_rows] = clip["body_positions"][frames, body_idx]
                root_quat[target_rows] = clip["body_rotations"][frames, body_idx]
                root_lin_vel[target_rows] = clip["body_linear_velocities"][frames, body_idx]
                root_ang_vel[target_rows] = clip["body_angular_velocities"][frames, body_idx]
                joint_pos[target_rows] = clip["dof_positions"][frames][:, clip_joint_indexes]
                joint_vel[target_rows] = clip["dof_velocities"][frames][:, clip_joint_indexes]

        root_pos += env.scene.env_origins[env_ids]
        joint_pos = joint_pos[:, asset_cfg.joint_ids]
        joint_vel = joint_vel[:, asset_cfg.joint_ids]
        limits = asset.data.soft_joint_pos_limits.torch[env_ids][:, asset_cfg.joint_ids]
        joint_pos = torch.maximum(torch.minimum(joint_pos, limits[..., 1]), limits[..., 0])
        asset.write_root_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.cat((root_lin_vel, root_ang_vel), dim=-1), env_ids=env_ids)
        asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def init_motion_loader(
    env: ManagerBasedRLEnv,
    env_ids,
    motion_dir: str,
    recovery_motion_names: tuple[str, ...] = (),
    delay_reset_env_ratio: float = 0.0,
    max_delay_steps: int = 0,
):
    """Load motion clips and configure optional recovery resets.

    Args:
        env: Environment that will use the motion clips.
        env_ids: Unused for this startup event.
        motion_dir: Directory containing the motion NPZ files.
        recovery_motion_names: Clip names reserved for recovery resets.
        delay_reset_env_ratio: Fraction of environments with delayed failure termination.
        max_delay_steps: Maximum delay before terminating those environments.
    """
    del env_ids
    MotionResetManager.init(motion_dir, str(env.device), recovery_motion_names)
    num_delay = int(env.num_envs * delay_reset_env_ratio)
    if num_delay and max_delay_steps > 0:
        delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        delay_mask[torch.randperm(env.num_envs, device=env.device)[:num_delay]] = True
        env.termination_manager = DelayedTerminationManager(env.termination_manager, delay_mask, max_delay_steps)


def reset_from_motion_data(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    root_name: str,
    asset_cfg: SceneEntityCfg,
    recovery_motion_names: tuple[str, ...] = (),
):
    """Reset from cached motion data, loading the clips first if needed."""
    MotionResetManager.reset(env, env_ids, motion_dir, root_name, asset_cfg, recovery_motion_names)


class RecoveryAssistForce(ManagerTermBase):
    """Apply a linearly decaying world-up force at one link's center of mass.

    Run as a global interval event every control step. The permanent wrench persists through
    all physics substeps. Event-manager resets restore it before the next physics step, after
    the articulation has cleared its wrench buffers and motion reset has selected the clip.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not math.isfinite(cfg.params["initial_force"]) or cfg.params["initial_force"] < 0.0:
            raise ValueError("Recovery assistance initial_force must be finite and non-negative (N)")
        if cfg.params["decay_steps"] <= 0:
            raise ValueError("Recovery assistance decay_steps must be positive (control steps)")
        asset_cfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._body_ids = asset_cfg.body_ids
        if isinstance(self._body_ids, slice):
            self._body_ids = list(range(self._asset.num_bodies))[self._body_ids]
        if len(self._body_ids) != 1:
            raise ValueError("Recovery assistance must select exactly one link")
        self._forces = torch.zeros((env.num_envs, 1, 3), device=env.device)
        self.current_force = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice | None,
        initial_force: float,
        decay_steps: int,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        """Refresh recovery-only forces; initial_force is in N and decay_steps in control steps."""
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
        self.current_force = initial_force * max(0.0, 1.0 - env.common_step_counter / decay_steps)
        self._forces[env_ids] = 0.0
        recovery_mask = getattr(env, "amp_recovery_env_mask", None)
        if recovery_mask is not None:
            self._forces[env_ids, 0, 2] = recovery_mask[env_ids] * self.current_force
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._forces[env_ids],
            body_ids=self._body_ids,
            env_ids=env_ids,
            is_global=True,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Restore the scheduled assistance on reset environments without restarting its decay."""
        self(self._env, env_ids, **self.cfg.params)


class DelayedTerminationManager(TerminationManager):
    """Keep selected environments alive briefly after a failure for recovery motions."""

    def __init__(self, base: TerminationManager, delay_env_mask: torch.Tensor, max_delay_steps: int):
        self.__dict__.update(base.__dict__)
        self._delay_env_mask = delay_env_mask
        self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
        self._max_delay_steps = max_delay_steps

    def compute(self) -> torch.Tensor:
        dones = super().compute()
        delayed_done = self._delay_env_mask & dones
        self._delay_counters[delayed_done] += 1
        pending = delayed_done & (self._delay_counters < self._max_delay_steps)
        self._terminated_buf[pending] = False
        ready = delayed_done & (self._delay_counters >= self._max_delay_steps)
        self._delay_counters[ready] = 0
        self._delay_counters[self._delay_env_mask & ~dones] = 0
        return self._truncated_buf | self._terminated_buf

    def reset(self, env_ids=None):
        if env_ids is None:
            self._delay_counters.zero_()
        else:
            self._delay_counters[env_ids] = 0
        return super().reset(env_ids)
