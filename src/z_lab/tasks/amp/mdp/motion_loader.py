# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load NPZ motion states and sample frames for robot resets, independently of RSL-RL."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

@dataclass
class FramePool:
    """Clips concatenated along the frame axis, in the loader's body and joint order.

    Positions and velocities use the recorded world frame, with positions in meters and velocities
    in meters/second or radians/second. Quaternions use ``xyzw``. No feature transforms, history
    stacking, scaling, or default joint offsets are applied.

    Attributes:
        body_pos_w: Body positions, shape ``(total_frames, num_bodies, 3)``.
        body_quat_w: Body orientations, shape ``(total_frames, num_bodies, 4)``.
        body_lin_vel_w: Body linear velocities, shape ``(total_frames, num_bodies, 3)``.
        body_ang_vel_w: Body angular velocities, shape ``(total_frames, num_bodies, 3)``.
        dof_pos: Joint positions, shape ``(total_frames, num_joints)``.
        dof_vel: Joint velocities, shape ``(total_frames, num_joints)``.
        counts: Frame count per clip, shape ``(num_clips,)``.
        fps: Sampling frequency [Hz] per clip, shape ``(num_clips,)``.
        motion_names: Clip names in the same order as ``counts`` and ``fps``.
    """

    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    counts: torch.Tensor
    fps: torch.Tensor
    motion_names: tuple[str, ...]

    @property
    def num_frames(self) -> int:
        """Total number of frames in the pool."""
        return self.body_pos_w.shape[0]

    @property
    def offsets(self) -> torch.Tensor:
        """Start frame of each clip, shape ``(num_clips,)``."""
        return self.counts.cumsum(0) - self.counts


class MotionLoader:
    """Load normal and recovery clips once, aligning tensors by body and joint names.

    The NPZ schema is the output of ``scripts/mimic/csv_to_npz.py``. All requested bodies and
    joints must exist in every clip. The root is identified using ``root_link_name``, independently
    of the anchor used for AMP features. Raw pools remain available as ``normal`` and ``recovery``.

    Args:
        motion_dir: Directory containing normal clips. With no separate recovery directory,
            filenames matching ``recovery_name_patterns`` are separated into the recovery pool.
        body_names: All robot bodies in simulation order, including the root.
        joint_names: All robot joints in simulation order.
        device: Device on which the pools and sampled states are stored.
        root_name: Expected articulation root. If omitted, use the first clip's root metadata.
        recovery_dir: Optional separate directory containing recovery clips.
        recovery_name_patterns: Case-insensitive filename substrings identifying recovery clips.
    """

    def __init__(
        self,
        motion_dir: str,
        body_names: Sequence[str],
        joint_names: Sequence[str],
        device: str | torch.device = "cpu",
        root_name: str | None = None,
        recovery_dir: str | None = None,
        recovery_name_patterns: Sequence[str] = ("fall", "getup", "get_up", "recovery"),
    ) -> None:
        self.device = torch.device(device)
        self.body_names = tuple(body_names)
        self.dof_names = tuple(joint_names)
        self.root_name = root_name
        for label, names in (("body_names", self.body_names), ("joint_names", self.dof_names)):
            if not names or len(set(names)) != len(names):
                raise ValueError(f"{label} must contain unique, nonempty names.")
        if root_name is not None and root_name not in self.body_names:
            raise ValueError(f"Root {root_name!r} is not in body_names.")

        motion_path = Path(motion_dir).expanduser().resolve()
        recovery_path = Path(recovery_dir).expanduser().resolve() if recovery_dir is not None else motion_path
        for directory in {motion_path, recovery_path}:
            if not directory.is_dir():
                raise FileNotFoundError(f"AMP motion directory does not exist: {directory}")

        normal_paths: list[Path] = []
        recovery_paths: list[Path] = []
        patterns = tuple(pattern.casefold() for pattern in recovery_name_patterns)
        if any(not pattern for pattern in patterns):
            raise ValueError("Recovery filename patterns must not be empty strings.")
        for path in sorted(motion_path.glob("*.npz")):
            is_recovery = recovery_path == motion_path and any(pattern in path.stem.casefold() for pattern in patterns)
            (recovery_paths if is_recovery else normal_paths).append(path)
        if recovery_path != motion_path:
            recovery_paths = sorted(recovery_path.glob("*.npz"))
            if not recovery_paths:
                raise FileNotFoundError(f"No recovery AMP motion files found in: {recovery_path}")
        if not normal_paths:
            raise FileNotFoundError(f"No normal AMP motion files found in: {motion_path}")

        self.normal = self._build_pool(normal_paths)
        self.recovery = self._build_pool(recovery_paths)
        self.root_body_id = self.body_names.index(self.root_name)

    @torch.no_grad()
    def sample_reset_state(self, num_envs: int, use_recovery: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Sample complete recorded states uniformly over frames within each selected pool.

        Args:
            num_envs: Number of states to sample.
            use_recovery: Boolean mask of shape ``(num_envs,)``. Marked rows use recovery clips,
                falling back to normal clips if the recovery pool is empty. Defaults to normal clips.

        Returns:
            Independent tensors keyed by ``root_pos``, ``root_quat``, ``root_lin_vel``,
            ``root_ang_vel``, ``dof_pos``, and ``dof_vel``. Each has ``num_envs`` rows. All fields
            in a row come from the same recorded frame. Poses have no environment-origin offset.
        """
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}.")
        if use_recovery is not None:
            if use_recovery.dtype != torch.bool or use_recovery.shape != (num_envs,):
                raise ValueError(f"use_recovery must be a bool tensor with shape ({num_envs},).")
            use_recovery = use_recovery.to(self.device)
        if use_recovery is None or self.recovery.num_frames == 0:
            use_recovery = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        state = {
            name: torch.empty(num_envs, width, dtype=torch.float32, device=self.device)
            for name, width in (
                ("root_pos", 3),
                ("root_quat", 4),
                ("root_lin_vel", 3),
                ("root_ang_vel", 3),
                ("dof_pos", len(self.dof_names)),
                ("dof_vel", len(self.dof_names)),
            )
        }
        for selected, pool in ((~use_recovery, self.normal), (use_recovery, self.recovery)):
            count = int(selected.sum())
            if count == 0:
                continue
            frame_ids = torch.randint(pool.num_frames, (count,), device=self.device)
            state["root_pos"][selected] = pool.body_pos_w[frame_ids, self.root_body_id]
            state["root_quat"][selected] = pool.body_quat_w[frame_ids, self.root_body_id]
            state["root_lin_vel"][selected] = pool.body_lin_vel_w[frame_ids, self.root_body_id]
            state["root_ang_vel"][selected] = pool.body_ang_vel_w[frame_ids, self.root_body_id]
            state["dof_pos"][selected] = pool.dof_pos[frame_ids]
            state["dof_vel"][selected] = pool.dof_vel[frame_ids]
        return state

    def _build_pool(self, paths: Sequence[Path]) -> FramePool:
        """Validate clips and concatenate raw tensors while preserving their frame boundaries."""
        fields = {
            "body_pos_w": ("body_positions", (len(self.body_names), 3)),
            "body_quat_w": ("body_rotations", (len(self.body_names), 4)),
            "body_lin_vel_w": ("body_linear_velocities", (len(self.body_names), 3)),
            "body_ang_vel_w": ("body_angular_velocities", (len(self.body_names), 3)),
            "dof_pos": ("dof_positions", (len(self.dof_names),)),
            "dof_vel": ("dof_velocities", (len(self.dof_names),)),
        }
        tensors: dict[str, list[torch.Tensor]] = {name: [] for name in fields}
        counts: list[int] = []
        frequencies: list[float] = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                required = {"body_names", "dof_names", "root_link_name", "fps"} | {key for key, _ in fields.values()}
                missing = required - set(data.files)
                if missing:
                    raise ValueError(f"Clip {path.name} is missing fields: {sorted(missing)}")
                clip_bodies = tuple(str(name) for name in data["body_names"])
                clip_joints = tuple(str(name) for name in data["dof_names"])
                for label, stored, requested in (
                    ("bodies", clip_bodies, self.body_names),
                    ("joints", clip_joints, self.dof_names),
                ):
                    if len(set(stored)) != len(stored):
                        raise ValueError(f"Clip {path.name} has duplicate {label}.")
                    missing_names = set(requested) - set(stored)
                    if missing_names:
                        raise ValueError(f"Clip {path.name} is missing {label}: {sorted(missing_names)}")
                root = data["root_link_name"]
                if root.size != 1:
                    raise ValueError(f"Clip {path.name} must specify exactly one root_link_name.")
                root_name = str(root.reshape(-1)[0])
                if root_name not in self.body_names:
                    raise ValueError(f"Clip {path.name} root {root_name!r} is not in body_names.")
                if self.root_name is None:
                    self.root_name = root_name
                elif root_name != self.root_name:
                    raise ValueError(f"Clip {path.name} root {root_name!r} differs from {self.root_name!r}.")
                fps = data["fps"]
                if fps.size != 1 or not np.isfinite(fps).all() or float(fps.reshape(-1)[0]) <= 0:
                    raise ValueError(f"Clip {path.name} must specify one finite, positive fps.")
                frequencies.append(float(fps.reshape(-1)[0]))
                num_frames = data["body_positions"].shape[0]
                if num_frames == 0:
                    raise ValueError(f"Clip {path.name} contains no frames.")
                counts.append(num_frames)
                body_ids = [clip_bodies.index(name) for name in self.body_names]
                joint_ids = [clip_joints.index(name) for name in self.dof_names]
                for name, (key, _) in fields.items():
                    values = data[key]
                    is_body = name.startswith("body_")
                    shape = (num_frames, len(clip_bodies), 4 if name == "body_quat_w" else 3)
                    if not is_body:
                        shape = (num_frames, len(clip_joints))
                    if values.shape != shape or not np.isfinite(values).all():
                        raise ValueError(f"Clip {path.name} field {key} must be finite with shape {shape}.")
                    values = values[:, body_ids if is_body else joint_ids]
                    tensors[name].append(torch.tensor(values, dtype=torch.float32, device=self.device))

        merged = {
            name: torch.cat(tensors[name], dim=0)
            if tensors[name]
            else torch.empty((0, *shape), dtype=torch.float32, device=self.device)
            for name, (_, shape) in fields.items()
        }
        return FramePool(
            **merged,
            counts=torch.tensor(counts, dtype=torch.long, device=self.device),
            fps=torch.tensor(frequencies, dtype=torch.float32, device=self.device),
            motion_names=tuple(path.stem for path in paths),
        )
