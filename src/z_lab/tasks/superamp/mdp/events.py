"""Reset locomotion environments from AMP expert frames."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg, SceneEntityCfg


class ResetFromExpertMotion(ManagerTermBase):
    """Preload expert root and joint states in articulation order for vectorized resets."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._robot = env.scene[cfg.params["asset_cfg"].name]
        motion_file = Path(cfg.params["motion_file"])
        paths = [motion_file] if motion_file.is_file() else sorted(motion_file.rglob("*.npz"))
        excluded = set(cfg.params["exclude_motion_names"])
        paths = [path for path in paths if path.stem not in excluded]
        if not paths:
            raise FileNotFoundError(f"No eligible expert motion NPZ files at {motion_file}")

        root_name = cfg.params["root_name"]
        self._joint_count = len(self._robot.joint_names)
        clips = []
        for path in paths:
            with np.load(path, allow_pickle=False) as motion:
                fps = float(motion["fps"].item())
                if not math.isclose(fps * env.step_dt, 1.0, rel_tol=1.0e-6):
                    raise ValueError(f"{path}: expert FPS {fps:g} does not match control FPS {1 / env.step_dt:g}")
                bodies = motion["body_names"].tolist()
                joints = motion["dof_names"].tolist()
                try:
                    root = bodies.index(root_name)
                    joint_ids = [joints.index(name) for name in self._robot.joint_names]
                except ValueError as exc:
                    raise ValueError(f"{path}: expert body or joint names do not match the robot: {exc}") from exc
                clip = np.concatenate(
                    (
                        motion["body_positions"][:, root],
                        motion["body_rotations"][:, root],
                        motion["body_linear_velocities"][:, root],
                        motion["body_angular_velocities"][:, root],
                        motion["dof_positions"][:, joint_ids],
                        motion["dof_velocities"][:, joint_ids],
                    ),
                    axis=-1,
                )
                if len(clip) < 2 or not np.isfinite(clip).all():
                    raise ValueError(f"{path}: expected at least two finite expert frames")
                clips.append(clip)
        self._frames = torch.as_tensor(np.concatenate(clips), device=env.device, dtype=torch.float32)
        # Remove accumulated travel in the recording while preserving the sampled pelvis height and twist.
        self._frames[:, :2] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        motion_file: str,
        root_name: str,
        exclude_motion_names: tuple[str, ...],
    ) -> None:
        """Write a random expert frame to each selected environment."""
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
        if env_ids.numel() == 0:
            return
        indices = torch.randint(len(self._frames), (env_ids.numel(),), device=env.device)
        frames = self._frames[indices]
        frames[:, :3] += env.scene.env_origins[env_ids]
        pose, twist, joint_pos, joint_vel = frames.split((7, 6, self._joint_count, self._joint_count), dim=-1)
        lower, upper = self._robot.data.soft_joint_pos_limits.torch[env_ids].unbind(dim=-1)
        self._robot.write_root_link_pose_to_sim_index(root_pose=pose.contiguous(), env_ids=env_ids)
        self._robot.write_root_link_velocity_to_sim_index(root_velocity=twist.contiguous(), env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos.clamp(lower, upper), env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel.contiguous(), env_ids=env_ids)
