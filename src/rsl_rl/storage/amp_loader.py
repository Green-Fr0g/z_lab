# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load expert AMP transitions and store policy transitions for replay sampling."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms


class AMPLoader:
    """Preload body-frame AMP transitions from NPZ files with xyzw body quaternions."""

    def __init__(
        self,
        motion_file: str | Path,
        body_names: tuple[str, ...] | list[str],
        anchor_name: str,
        device: str | torch.device = "cpu",
        exclude_motion_names: tuple[str, ...] | list[str] = (),
    ) -> None:
        """Load clips and precompute body-frame observations for every adjacent frame pair."""
        path = Path(motion_file)
        if not path.exists():
            raise FileNotFoundError(f"AMP motion path does not exist: {path}")
        files = [path] if path.is_file() else sorted(path.rglob("*.npz"))
        excluded = set(exclude_motion_names)
        files = [file for file in files if file.stem not in excluded]
        if not files:
            raise ValueError(f"No AMP motion NPZ files found under {path}")

        self.device = torch.device(device)
        self.motions: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.motion_names: list[str] = []
        for file in files:
            with np.load(file, allow_pickle=False) as data:
                required = (
                    "body_names",
                    "body_positions",
                    "body_rotations",
                    "body_linear_velocities",
                    "body_angular_velocities",
                )
                missing = [key for key in required if key not in data]
                if missing:
                    raise KeyError(f"{file} is missing AMP fields: {missing}")
                names = [str(name) for name in data["body_names"].tolist()]
                if anchor_name not in names:
                    raise ValueError(f"{file} has no anchor body {anchor_name!r}")
                missing_bodies = sorted(set(body_names) - set(names))
                if missing_bodies:
                    raise ValueError(f"{file} is missing AMP bodies: {missing_bodies}")
                body_ids = torch.tensor([names.index(name) for name in body_names], device=self.device)
                anchor_id = names.index(anchor_name)
                positions = torch.as_tensor(data["body_positions"], dtype=torch.float32, device=self.device)
                rotations = torch.as_tensor(data["body_rotations"], dtype=torch.float32, device=self.device)
                linear = torch.as_tensor(data["body_linear_velocities"], dtype=torch.float32, device=self.device)
                angular = torch.as_tensor(data["body_angular_velocities"], dtype=torch.float32, device=self.device)

            anchor_pos = positions[:, anchor_id : anchor_id + 1]
            anchor_rot = rotations[:, anchor_id : anchor_id + 1]
            body_pos = positions.index_select(1, body_ids)
            body_rot = rotations.index_select(1, body_ids)
            relative_pos, relative_rot = subtract_frame_transforms(
                anchor_pos.expand_as(body_pos), anchor_rot.expand_as(body_rot), body_pos, body_rot
            )
            relative_ori = matrix_from_quat(relative_rot)[..., :2].flatten(start_dim=1)
            body_linear = quat_apply_inverse(body_rot, linear.index_select(1, body_ids))
            body_angular = quat_apply_inverse(body_rot, angular.index_select(1, body_ids))
            observations = torch.cat(
                (
                    relative_pos.flatten(start_dim=1),
                    relative_ori,
                    body_linear.flatten(start_dim=1),
                    body_angular.flatten(start_dim=1),
                ),
                dim=-1,
            )
            if observations.shape[0] < 2:
                continue
            self.motions.append((observations[:-1].contiguous(), observations[1:].contiguous()))
            self.motion_names.append(file.stem)

        if not self.motions:
            raise ValueError(f"AMP motion files under {path} do not contain at least two frames")
        self.observation_dim = self.motions[0][0].shape[-1]
        if any(states.shape[-1] != self.observation_dim for states, _ in self.motions):
            raise ValueError("AMP motion files have inconsistent observation dimensions")

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample transitions uniformly across clips, then uniformly across their frames."""
        motion_ids = torch.randint(len(self.motions), (batch_size,), device=self.device)
        states = torch.empty(batch_size, self.observation_dim, device=self.device)
        next_states = torch.empty_like(states)
        for motion_id in motion_ids.unique().tolist():
            rows = torch.nonzero(motion_ids == motion_id, as_tuple=False).squeeze(-1)
            motion_states, motion_next_states = self.motions[motion_id]
            frame_ids = torch.randint(motion_states.shape[0], (rows.numel(),), device=self.device)
            states[rows] = motion_states[frame_ids]
            next_states[rows] = motion_next_states[frame_ids]
        return states, next_states

    def feed_forward_generator(
        self, num_mini_batches: int, mini_batch_size: int
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """Yield expert transition batches for optimizer updates."""
        for _ in range(num_mini_batches):
            yield self.sample(mini_batch_size)


class ReplayBuffer:
    """Fixed-capacity circular buffer for paired AMP observations."""

    def __init__(self, observation_dim: int, capacity: int, device: str | torch.device) -> None:
        """Allocate paired state buffers on the requested device."""
        if observation_dim <= 0 or capacity <= 0:
            raise ValueError("observation_dim and capacity must be positive")
        self.states = torch.empty(capacity, observation_dim, device=device)
        self.next_states = torch.empty_like(self.states)
        self.capacity = capacity
        self.position = 0
        self.size = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        """Insert a batch of paired states, retaining only the newest capacity rows."""
        states = states.reshape(-1, self.states.shape[-1]).to(self.states.device)
        next_states = next_states.reshape_as(states).to(self.next_states.device)
        if states.shape != next_states.shape:
            raise ValueError("states and next_states must have matching shapes")
        if states.shape[0] >= self.capacity:
            self.states.copy_(states[-self.capacity :])
            self.next_states.copy_(next_states[-self.capacity :])
            self.position = 0
            self.size = self.capacity
            return
        end = self.position + states.shape[0]
        first_count = min(states.shape[0], self.capacity - self.position)
        self.states[self.position : self.position + first_count] = states[:first_count]
        self.next_states[self.position : self.position + first_count] = next_states[:first_count]
        remaining = states.shape[0] - first_count
        if remaining:
            self.states[:remaining] = states[first_count:]
            self.next_states[:remaining] = next_states[first_count:]
        self.position = end % self.capacity
        self.size = min(self.capacity, self.size + states.shape[0])

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample paired rows with replacement from initialized entries."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty AMP replay buffer")
        indices = torch.randint(self.size, (batch_size,), device=self.states.device)
        return self.states[indices], self.next_states[indices]

    def feed_forward_generator(
        self, num_mini_batches: int, mini_batch_size: int
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """Yield replay batches for the requested number of optimizer updates."""
        for _ in range(num_mini_batches):
            yield self.sample(mini_batch_size)
