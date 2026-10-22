"""Expert motion loader for the AMP discriminator.

The loader turns recorded clips into ``(state, next_state)`` pairs using the same body features the
environment publishes in its ``amp_state`` observation group.

The NPZ schema is the one written by ``scripts/mimic/csv_to_npz.py``. Feature selection is driven
by names, so the recorded body order does not have to match the simulation order.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms

__all__ = ["AMPBodyStateMotionLoader"]

_FEATURE_DIM_PER_BODY = 3 + 6 + 3 + 3
"""Width contributed by one body: position, two rotation columns, linear and angular velocity."""


class AMPBodyStateMotionLoader:
    """Load AMP expert states from recorded body-state ``.npz`` motion files.

    Args:
        motion_files: A motion directory scanned recursively for ``*.npz`` files, a single clip, or
            an explicit sequence of clips.
        body_names: Bodies that contribute discriminator features, in feature order.
        anchor_name: Body whose frame defines relative positions and orientations.
        all_body_names: Every body that may appear in the clips. Recorded arrays are reordered to
            this layout before feature selection.
        device: Device holding the clips and the sampled batches.
        time_between_frames: Control period of the recorded clips. Accepted for interface
            compatibility with the training configuration; sampling is frame based.
        quat_order: Order of the recorded quaternions, ``"xyzw"`` (default) or ``"wxyz"``.
    """

    def __init__(
        self,
        motion_files: str | Sequence[str],
        body_names: Sequence[str],
        anchor_name: str,
        all_body_names: Sequence[str],
        device: str = "cpu",
        time_between_frames: float = 0.02,
        quat_order: str = "xyzw",
    ) -> None:
        del time_between_frames
        if quat_order not in ("xyzw", "wxyz"):
            raise ValueError(f"Unsupported quat_order '{quat_order}'. Expected 'xyzw' or 'wxyz'.")

        self.device = torch.device(device)
        self.body_names = tuple(body_names)
        self.anchor_name = anchor_name
        self.all_body_names = tuple(all_body_names)
        self.quat_order = quat_order
        for label, names in (("body_names", self.body_names), ("anchor_name", (self.anchor_name,))):
            missing = [name for name in names if name not in self.all_body_names]
            if missing:
                raise ValueError(f"AMP {label} {missing} are not part of all_body_names.")

        self._body_indexes = [self.all_body_names.index(name) for name in self.body_names]
        self._anchor_index = self.all_body_names.index(self.anchor_name)
        self._num_bodies = len(self._body_indexes)

        self.motion_files = self._resolve_motion_files(motion_files)
        self._states: list[torch.Tensor] = []
        self._next_states: list[torch.Tensor] = []
        for motion_file in self.motion_files:
            state = self._load_motion_state(motion_file)
            if state.shape[0] < 2:
                continue
            self._states.append(state[:-1])
            self._next_states.append(state[1:])

        if not self._states:
            raise RuntimeError("No AMP expert states were loaded.")

    @property
    def observation_dim(self) -> int:
        """Width of a single AMP state."""
        return _FEATURE_DIM_PER_BODY * self._num_bodies

    @property
    def state_dim(self) -> int:
        """Width of a single AMP state, as consumed by the discriminator."""
        return self.observation_dim

    def mini_batch_generator(self, num_mini_batch: int, mini_batch_size: int) -> Generator:
        """Yield ``(states, next_states)`` mini-batches cycling through the loaded clips."""
        for batch_idx in range(num_mini_batch):
            motion_idx = batch_idx % len(self._states)
            states = self._states[motion_idx]
            next_states = self._next_states[motion_idx]
            ids = torch.randint(0, states.shape[0], (mini_batch_size,), device=self.device)
            yield states[ids], next_states[ids]

    def _load_motion_state(self, motion_file: Path) -> torch.Tensor:
        """Read one clip and return its features in ``(num_frames, state_dim)`` layout."""
        with np.load(motion_file, allow_pickle=False) as data:
            for key in (
                "dof_positions",
                "body_positions",
                "body_rotations",
                "body_linear_velocities",
                "body_angular_velocities",
            ):
                if key not in data:
                    raise KeyError(f"AMP motion file '{motion_file}' is missing key '{key}'.")

            body_positions = torch.as_tensor(data["body_positions"], dtype=torch.float32, device=self.device)
            body_rotations = torch.as_tensor(data["body_rotations"], dtype=torch.float32, device=self.device)
            body_linear_velocities = torch.as_tensor(
                data["body_linear_velocities"], dtype=torch.float32, device=self.device
            )
            body_angular_velocities = torch.as_tensor(
                data["body_angular_velocities"], dtype=torch.float32, device=self.device
            )

            if "body_names" not in data:
                raise KeyError(
                    f"AMP motion file '{motion_file}' is missing 'body_names'. "
                    "Re-export the motion with scripts/mimic/csv_to_npz.py."
                )
            motion_body_names = [str(name) for name in np.asarray(data["body_names"]).tolist()]
            if len(motion_body_names) != body_positions.shape[1]:
                raise ValueError(
                    f"AMP motion file '{motion_file}' has {len(motion_body_names)} body_names "
                    f"but body_positions has {body_positions.shape[1]} bodies."
                )

        missing_bodies = [name for name in self.all_body_names if name not in motion_body_names]
        if missing_bodies:
            raise ValueError(f"AMP motion file '{motion_file}' is missing bodies: {missing_bodies}")

        body_indexes = [motion_body_names.index(name) for name in self.all_body_names]
        body_positions = body_positions[:, body_indexes]
        body_rotations = body_rotations[:, body_indexes]
        body_linear_velocities = body_linear_velocities[:, body_indexes]
        body_angular_velocities = body_angular_velocities[:, body_indexes]
        if self.quat_order == "wxyz":
            body_rotations = body_rotations[..., [1, 2, 3, 0]]

        anchor_pos_w = body_positions[:, self._anchor_index]
        anchor_quat_w = body_rotations[:, self._anchor_index]
        target_pos_w = body_positions[:, self._body_indexes]
        target_quat_w = body_rotations[:, self._body_indexes]
        target_lin_vel_w = body_linear_velocities[:, self._body_indexes]
        target_ang_vel_w = body_angular_velocities[:, self._body_indexes]

        num_frames = target_pos_w.shape[0]
        anchor_pos_w = anchor_pos_w[:, None, :].expand(-1, self._num_bodies, -1)
        anchor_quat_w = anchor_quat_w[:, None, :].expand(-1, self._num_bodies, -1)
        body_pos_b, body_quat_b = subtract_frame_transforms(anchor_pos_w, anchor_quat_w, target_pos_w, target_quat_w)
        body_ori_b = matrix_from_quat(body_quat_b)[..., :2].reshape(num_frames, self._num_bodies, 6)
        body_lin_vel_b = quat_apply_inverse(target_quat_w.reshape(-1, 4), target_lin_vel_w.reshape(-1, 3)).reshape(
            num_frames, self._num_bodies, 3
        )
        body_ang_vel_b = quat_apply_inverse(target_quat_w.reshape(-1, 4), target_ang_vel_w.reshape(-1, 3)).reshape(
            num_frames, self._num_bodies, 3
        )

        return torch.cat(
            (
                body_pos_b.reshape(num_frames, -1),
                body_ori_b.reshape(num_frames, -1),
                body_lin_vel_b.reshape(num_frames, -1),
                body_ang_vel_b.reshape(num_frames, -1),
            ),
            dim=-1,
        )

    @staticmethod
    def _resolve_motion_files(motion_files: str | Sequence[str]) -> list[Path]:
        paths = [Path(motion_files)] if isinstance(motion_files, (str, Path)) else [Path(path) for path in motion_files]

        resolved: list[Path] = []
        for path in paths:
            path = path.expanduser().resolve()
            if path.is_file():
                if path.suffix == ".npz":
                    resolved.append(path)
            elif path.is_dir():
                resolved.extend(sorted(path.rglob("*.npz")))
            else:
                raise FileNotFoundError(f"AMP motion path does not exist: {path}")

        if not resolved:
            raise FileNotFoundError(f"No AMP motion .npz files found in: {motion_files}")
        return sorted(resolved)
