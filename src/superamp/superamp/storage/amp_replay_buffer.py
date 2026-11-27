"""Expert and simulator AMP windows with a configurable frame stride."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat

from ..amp import body_amp_features


class ExpertTransitions:
    """Precompute ordered AMP features from control-rate link-origin NPZ clips."""

    def __init__(
        self,
        motion_path: str | Path,
        body_names: Sequence[str],
        anchor_name: str,
        *,
        control_dt: float,
        num_frames: int = 2,
        frame_stride: int = 1,
        device: str | torch.device = "cpu",
        exclude_motion_names: Sequence[str] = (),
    ) -> None:
        if not body_names or len(set(body_names)) != len(body_names):
            raise ValueError("AMP body names must be nonempty and unique")
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        path = Path(motion_path)
        if not path.exists():
            raise FileNotFoundError(path)
        paths = [path] if path.is_file() else sorted(path.rglob("*.npz"))
        excluded = set(exclude_motion_names)
        self.device = torch.device(device)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.window_length = _window_length(num_frames, frame_stride)
        self.motion_names: list[str] = []
        clips = []
        for clip_path in paths:
            if clip_path.stem in excluded:
                continue
            clip = self._load_clip(clip_path, body_names, anchor_name, control_dt)
            if len(clip) >= self.window_length:
                clips.append(clip)
                self.motion_names.append(clip_path.stem)
        if not clips:
            raise ValueError(f"No expert clip with at least {self.window_length} frames found at {path}")

        lengths = torch.tensor([len(clip) for clip in clips], device=self.device)
        self._starts = lengths.cumsum(dim=0) - lengths
        self._start_counts = lengths - self.window_length + 1
        self._frame_offsets = torch.arange(num_frames, device=self.device) * frame_stride
        self._features = torch.cat(clips)
        self.observation_dim = self._features.shape[-1]

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample windows of shape (batch_size, num_frames, observation_dim) within clips."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        clips = torch.randint(len(self.motion_names), (batch_size,), device=self.device)
        offsets = (torch.rand(batch_size, device=self.device) * self._start_counts[clips]).long()
        indices = self._starts[clips] + offsets
        return self._features[indices[:, None] + self._frame_offsets]

    def _load_clip(self, path: Path, body_names: Sequence[str], anchor_name: str, control_dt: float) -> torch.Tensor:
        with np.load(path, allow_pickle=False) as motion:
            fps = float(motion["fps"].item())
            if not math.isclose(fps * control_dt, 1.0, rel_tol=1.0e-6):
                raise ValueError(f"{path}: expert FPS {fps:g} does not match control FPS {1 / control_dt:g}")
            names = motion["body_names"].tolist()
            try:
                body_ids = [names.index(name) for name in body_names]
                anchor_id = names.index(anchor_name)
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from error

            positions = torch.as_tensor(motion["body_positions"], dtype=torch.float32, device=self.device)
            quaternions = torch.as_tensor(motion["body_rotations"], dtype=torch.float32, device=self.device)
            linear = torch.as_tensor(
                motion["body_linear_velocities"][:, body_ids], dtype=torch.float32, device=self.device
            )
            angular = torch.as_tensor(
                motion["body_angular_velocities"][:, body_ids], dtype=torch.float32, device=self.device
            )

        return body_amp_features(
            positions[:, anchor_id],
            matrix_from_quat(quaternions[:, anchor_id]),
            positions[:, body_ids],
            matrix_from_quat(quaternions[:, body_ids]),
            linear,
            angular,
        )


class AMPReplayBuffer:
    """Store strided simulator AMP windows in a fixed-capacity ring."""

    def __init__(
        self,
        observation_dim: int,
        capacity: int,
        device: str | torch.device = "cpu",
        *,
        num_frames: int = 2,
        frame_stride: int = 1,
    ) -> None:
        if observation_dim <= 0 or capacity <= 0:
            raise ValueError("observation_dim and capacity must be positive")
        self.observation_dim = observation_dim
        self.capacity = capacity
        self.device = torch.device(device)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.window_length = _window_length(num_frames, frame_stride)
        self.size = 0
        self._position = 0
        self._windows = torch.empty(capacity, num_frames, observation_dim, device=self.device)

    @torch.no_grad()
    def insert(self, windows: torch.Tensor) -> None:
        """Store windows from consecutive control steps within individual episodes.

        Input shape is (batch_size, window_length, observation_dim). A window
        ending at a terminal transition uses the true final frame before reset.
        """
        expected_shape = (self.window_length, self.observation_dim)
        if windows.ndim != 3 or windows.shape[1:] != expected_shape:
            raise ValueError(f"AMP windows must have shape (batch_size, {expected_shape[0]}, {expected_shape[1]})")
        windows = windows[-self.capacity :, :: self.frame_stride]
        count = len(windows)
        first_count = min(count, self.capacity - self._position)
        end = self._position + first_count
        self._windows[self._position : end].copy_(windows[:first_count])
        self._windows[: count - first_count].copy_(windows[first_count:])
        self._position = (self._position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample stored windows of shape (batch_size, num_frames, observation_dim)."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty AMP replay buffer")
        indices = torch.randint(self.size, (batch_size,), device=self.device)
        return self._windows[indices]


def _window_length(num_frames: int, frame_stride: int) -> int:
    if num_frames < 2 or frame_stride < 1:
        raise ValueError("AMP windows require num_frames >= 2 and frame_stride >= 1")
    return 1 + (num_frames - 1) * frame_stride
