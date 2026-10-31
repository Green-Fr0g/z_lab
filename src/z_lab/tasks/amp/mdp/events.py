"""Motion loading and motion-based resets for AMP tasks.

Recorded clips are cached once per motion directory and shared by every environment. Resets sample
a clip and a frame uniformly, write the root pose/velocity and the joint state, and clamp joint
positions to the articulation's soft limits.

The NPZ schema is the one written by ``scripts/mimic/csv_to_npz.py``. Bodies and joints are aligned
by name, so the recorded order does not have to match the simulation order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["MotionResetManager", "init_motion_loader", "reset_from_motion_data"]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot", joint_names=".*")


def _as_str_list(values) -> list[str]:
    return [str(name) for name in np.asarray(values).tolist()]


def _index_by_names(available: tuple[str, ...], requested: tuple[str, ...] | list[str], kind: str) -> list[int]:
    lookup = {name: index for index, name in enumerate(available)}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise ValueError(f"AMP motion {kind} names {missing} were not found. Available {kind} names: {list(available)}")
    return [lookup[name] for name in requested]


@dataclass
class _MotionClip:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_names: tuple[str, ...]

    @property
    def num_frames(self) -> int:
        return self.root_pos.shape[0]


class MotionResetManager:
    """Cache AMP motion frames and reset environments from sampled frames."""

    _instance: MotionResetManager | None = None

    def __init__(self) -> None:
        self._frames: dict[str, list[_MotionClip]] = {}

    @classmethod
    def get(cls) -> MotionResetManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init(
        self,
        motion_dir: str,
        device: str | torch.device,
        root_name: str,
        all_body_names: tuple[str, ...],
    ) -> None:
        """Load every clip once for the given motion directory, root, and body layout."""
        motion_dir = str(Path(motion_dir).expanduser().resolve())
        if root_name not in all_body_names:
            raise ValueError(f"AMP root body '{root_name}' is not in all_body_names.")
        root_index = all_body_names.index(root_name)
        cache_key = f"{motion_dir}:{root_index}:{tuple(all_body_names)}"
        if cache_key in self._frames:
            return

        files = self._collect_motion_files(motion_dir)
        if not files:
            raise FileNotFoundError(f"No AMP motion .npz files found in: {motion_dir}")
        clips: list[_MotionClip] = []
        for file in files:
            data = np.load(file)
            for key in (
                "body_names",
                "dof_names",
                "body_positions",
                "body_rotations",
                "body_linear_velocities",
                "body_angular_velocities",
                "dof_positions",
                "dof_velocities",
            ):
                if key not in data:
                    raise KeyError(f"AMP motion file '{file}' is missing key '{key}'.")

            body_positions = torch.as_tensor(data["body_positions"], dtype=torch.float32, device=device)
            body_rotations = torch.as_tensor(data["body_rotations"], dtype=torch.float32, device=device)
            body_linear_velocities = torch.as_tensor(data["body_linear_velocities"], dtype=torch.float32, device=device)
            body_angular_velocities = torch.as_tensor(
                data["body_angular_velocities"], dtype=torch.float32, device=device
            )
            dof_positions = torch.as_tensor(data["dof_positions"], dtype=torch.float32, device=device)
            dof_velocities = torch.as_tensor(data["dof_velocities"], dtype=torch.float32, device=device)

            motion_body_names = _as_str_list(data["body_names"])
            if len(motion_body_names) != body_positions.shape[1]:
                raise ValueError(
                    f"AMP motion file '{file}' has {len(motion_body_names)} body_names "
                    f"but body_positions has {body_positions.shape[1]} bodies."
                )
            body_indexes = _index_by_names(tuple(motion_body_names), all_body_names, "body")
            body_positions = body_positions[:, body_indexes]
            body_rotations = body_rotations[:, body_indexes]
            body_linear_velocities = body_linear_velocities[:, body_indexes]
            body_angular_velocities = body_angular_velocities[:, body_indexes]

            joint_names = _as_str_list(data["dof_names"])
            if len(joint_names) != dof_positions.shape[1]:
                raise ValueError(
                    f"AMP motion file '{file}' has {len(joint_names)} dof_names "
                    f"but dof_positions has {dof_positions.shape[1]} joints."
                )

            clips.append(
                _MotionClip(
                    root_pos=body_positions[:, root_index, :],
                    root_quat=body_rotations[:, root_index, :],
                    root_lin_vel=body_linear_velocities[:, root_index, :],
                    root_ang_vel=body_angular_velocities[:, root_index, :],
                    joint_pos=dof_positions,
                    joint_vel=dof_velocities,
                    joint_names=tuple(joint_names),
                )
            )

        self._frames[cache_key] = clips

    def reset(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        motion_dir: str,
        root_name: str,
        all_body_names: tuple[str, ...],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        """Write a randomly sampled recorded frame into the selected environments."""
        motion_dir = str(Path(motion_dir).expanduser().resolve())
        cache_key = f"{motion_dir}:{all_body_names.index(root_name)}:{tuple(all_body_names)}"
        if cache_key not in self._frames:
            self.init(motion_dir, env.device, root_name, all_body_names)

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        if len(env_ids) == 0:
            return

        clips = self._frames[cache_key]
        clip_ids = torch.randint(0, len(clips), (len(env_ids),), device=env.device)
        frame_ids = torch.zeros_like(clip_ids)
        for clip_id in clip_ids.unique().tolist():
            selected = clip_ids == clip_id
            frame_ids[selected] = torch.randint(
                clips[clip_id].num_frames, (int(selected.sum().item()),), device=env.device
            )

        asset: Articulation = env.scene[asset_cfg.name]
        asset_joint_names = tuple(asset.joint_names)
        if not asset_joint_names:
            raise RuntimeError("AMP asset does not expose joint_names for motion alignment.")

        root_pos = torch.empty((len(env_ids), 3), device=env.device, dtype=torch.float32)
        root_quat = torch.empty((len(env_ids), 4), device=env.device, dtype=torch.float32)
        root_lin_vel = torch.empty((len(env_ids), 3), device=env.device, dtype=torch.float32)
        root_ang_vel = torch.empty((len(env_ids), 3), device=env.device, dtype=torch.float32)
        joint_pos = torch.empty((len(env_ids), len(asset_joint_names)), device=env.device, dtype=torch.float32)
        joint_vel = torch.empty_like(joint_pos)

        for clip_id in clip_ids.unique().tolist():
            selected = clip_ids == clip_id
            clip = clips[clip_id]
            clip_frame_ids = frame_ids[selected]
            root_pos[selected] = clip.root_pos[clip_frame_ids]
            root_quat[selected] = clip.root_quat[clip_frame_ids]
            root_lin_vel[selected] = clip.root_lin_vel[clip_frame_ids]
            root_ang_vel[selected] = clip.root_ang_vel[clip_frame_ids]

            clip_joint_pos = clip.joint_pos[clip_frame_ids]
            clip_joint_vel = clip.joint_vel[clip_frame_ids]
            joint_indexes = _index_by_names(clip.joint_names, asset_joint_names, "joint")
            joint_pos[selected] = clip_joint_pos[:, joint_indexes]
            joint_vel[selected] = clip_joint_vel[:, joint_indexes]

        root_pos = root_pos.clone()
        root_pos[:, :2] += env.scene.env_origins[env_ids, :2]
        root_pos[:, 2] += env.scene.env_origins[env_ids, 2]
        root_pose = torch.cat((root_pos, root_quat), dim=-1)
        root_velocity = torch.cat((root_lin_vel, root_ang_vel), dim=-1)

        joint_pos = joint_pos[:, asset_cfg.joint_ids]
        joint_vel = joint_vel[:, asset_cfg.joint_ids]
        joint_limits = asset.data.soft_joint_pos_limits.torch[env_ids][:, asset_cfg.joint_ids]
        joint_pos = joint_pos.clamp(joint_limits[..., 0], joint_limits[..., 1])

        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

    @staticmethod
    def _collect_motion_files(motion_dir: str) -> list[Path]:
        path = Path(motion_dir)
        if path.is_file() and path.suffix == ".npz":
            return [path]
        return sorted(path.rglob("*.npz"))


def init_motion_loader(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    root_name: str,
    all_body_names: tuple[str, ...],
) -> None:
    """Load AMP motion data during startup."""
    del env_ids
    MotionResetManager.get().init(motion_dir, env.device, root_name, all_body_names)


def reset_from_motion_data(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    root_name: str,
    all_body_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset selected environments from random AMP motion frames."""
    MotionResetManager.get().reset(env, env_ids, motion_dir, root_name, all_body_names, asset_cfg)
