# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent NPZ loading, episode-aware policy collection, and shared AMP minibatches."""

from __future__ import annotations

import math
from collections.abc import Generator, Sequence
from pathlib import Path

import numpy as np
import torch

from .amp_features import DEFAULT_OBS_STRING, AmpFeatureEncoder


class AmpDataset:
    """Own expert clips, raw policy history, and one rollout of discriminator inputs.

    Raw observations pack all bodies/joints in simulation order: world position, xyzw orientation,
    world linear/angular velocity, absolute joint position, absolute joint velocity. Each field is
    flattened before concatenation. The shared encoder selects names and generates invariant features.

    ``num_frames`` and ``history_stride`` define an oldest-to-newest window. Stride counts control
    steps; expert fps must equal ``1 / step_dt``. Experts use only complete clip windows. Policy
    histories repeat the first episode frame until enough real frames have been collected.

    Recovery contributes ``int(batch_size * recovery_sample_ratio)`` samples per expert batch,
    falling back to normal clips if no recovery clips exist. Within each pool, choose eligible files
    uniformly, then valid starts uniformly. Every minibatch draws new expert windows.

    With ``replay_buffer_size > 0``, policy batches are sampled with replacement from a persistent
    ring buffer. Batch size follows the current PPO rollout, independently of replay capacity.
    Zero uses shuffled current-rollout samples without replacement. Raw episode history survives
    :meth:`clear_rollout` in both modes.

    Args:
        motion_dir: Directory of NPZ clips, using the schema from ``csv_to_npz.py``.
        policy_body_names: Full body order of the simulation observation producer.
        policy_joint_names: Full joint order of the simulation observation producer.
        anchor_name: Body used for relative positions and orientations, independently of the root.
        num_envs: Policy environments collected at every step.
        num_steps_per_env: Capacity of one discriminator rollout.
        step_dt: Environment control period in seconds.
        body_names: Ordered feature bodies; defaults to all policy bodies.
        joint_names: Ordered feature joints; defaults to all policy joints.
        obs_string: Ordered registered feature names, shared by expert and policy.
        recovery_dir: Optional explicit recovery directory. Otherwise classify filenames by patterns.
        recovery_name_patterns: Case-insensitive recovery filename substrings.
        num_frames: Number of frames per discriminator input.
        history_stride: Control steps between window entries.
        recovery_sample_ratio: Recovery fraction per expert batch.
        replay_buffer_size: Maximum retained policy windows; zero disables replay.
        default_joint_pos: Robot defaults in full simulation joint order, required for joint_pos.
        default_joint_vel: Robot defaults in full simulation joint order, required for joint_vel.
        joint_pos_scale: Shared relative joint position multiplier.
        joint_vel_scale: Shared relative joint velocity multiplier.
        device: Device for expert clips, policy history, and training samples.
    """

    def __init__(
        self,
        motion_dir: str,
        policy_body_names: Sequence[str],
        policy_joint_names: Sequence[str],
        anchor_name: str,
        *,
        num_envs: int,
        num_steps_per_env: int,
        step_dt: float,
        body_names: Sequence[str] | None = None,
        joint_names: Sequence[str] | None = None,
        obs_string: Sequence[str] = DEFAULT_OBS_STRING,
        recovery_dir: str | None = None,
        recovery_name_patterns: Sequence[str] = ("fall", "getup", "get_up", "recovery"),
        num_frames: int = 2,
        history_stride: int = 1,
        recovery_sample_ratio: float = 0.2,
        replay_buffer_size: int = 0,
        default_joint_pos: torch.Tensor | None = None,
        default_joint_vel: torch.Tensor | None = None,
        joint_pos_scale: float = 1.0,
        joint_vel_scale: float = 0.05,
        device: str | torch.device = "cpu",
    ) -> None:
        for name, value in (
            ("num_envs", num_envs),
            ("num_steps_per_env", num_steps_per_env),
            ("num_frames", num_frames),
            ("history_stride", history_stride),
        ):
            self._check_positive_int(name, value)
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("step_dt must be finite and positive.")
        if not 0 <= recovery_sample_ratio <= 1:
            raise ValueError("recovery_sample_ratio must be in [0, 1].")
        if isinstance(replay_buffer_size, bool) or not isinstance(replay_buffer_size, int) or replay_buffer_size < 0:
            raise ValueError(f"replay_buffer_size must be a nonnegative integer, got {replay_buffer_size!r}.")
        self.device = torch.device(device)
        self.num_envs, self.num_steps_per_env = num_envs, num_steps_per_env
        self.num_frames, self.history_stride = num_frames, history_stride
        self.recovery_sample_ratio, self.step_dt = recovery_sample_ratio, step_dt
        self.policy_body_names = tuple(policy_body_names)
        self.policy_joint_names = tuple(policy_joint_names)
        self.encoder = AmpFeatureEncoder(
            self.policy_body_names,
            self.policy_joint_names,
            self.policy_body_names if body_names is None else tuple(body_names),
            self.policy_joint_names if joint_names is None else tuple(joint_names),
            anchor_name,
            obs_string,
            default_joint_pos,
            default_joint_vel,
            joint_pos_scale,
            joint_vel_scale,
            self.device,
        )
        nb, nj = len(self.policy_body_names), len(self.policy_joint_names)
        self.raw_shapes = {
            "body_pos_w": (nb, 3),
            "body_quat_w": (nb, 4),
            "body_lin_vel_w": (nb, 3),
            "body_ang_vel_w": (nb, 3),
            "dof_pos": (nj,),
            "dof_vel": (nj,),
        }
        self.raw_dim = sum(math.prod(shape) for shape in self.raw_shapes.values())
        self.frame_dim = self.encoder.frame_dim
        self.obs_dim = self.frame_dim * num_frames
        self._span = (num_frames - 1) * history_stride + 1
        self._frame_steps = torch.arange(num_frames, device=self.device) * history_stride
        self.replay_buffer_size = replay_buffer_size
        self._replay: torch.Tensor | None = None
        self._replay_cursor = 0
        self._replay_filled = 0
        if replay_buffer_size:
            self._replay = torch.empty(replay_buffer_size, self.obs_dim, device=self.device)

        motion_path = Path(motion_dir).expanduser().resolve()
        recovery_path = Path(recovery_dir).expanduser().resolve() if recovery_dir is not None else motion_path
        for directory in {motion_path, recovery_path}:
            if not directory.is_dir():
                raise FileNotFoundError(f"AMP motion directory does not exist: {directory}")
        patterns = tuple(pattern.casefold() for pattern in recovery_name_patterns)
        if any(not pattern for pattern in patterns):
            raise ValueError("Recovery filename patterns must not be empty.")
        normal_paths, recovery_paths = [], []
        for path in sorted(motion_path.glob("*.npz")):
            recovery = recovery_path == motion_path and any(p in path.stem.casefold() for p in patterns)
            (recovery_paths if recovery else normal_paths).append(path)
        if recovery_path != motion_path:
            recovery_paths = sorted(recovery_path.glob("*.npz"))
            if not recovery_paths:
                raise FileNotFoundError(f"No recovery motion clips found in {recovery_path}.")
        if not normal_paths:
            raise FileNotFoundError(f"No normal motion clips found in {motion_path}.")
        self.motion_names = tuple(path.stem for path in normal_paths)
        self.recovery_motion_names = tuple(path.stem for path in recovery_paths)
        self._root_name: str | None = None
        self._normal, counts = self._load_clips(normal_paths)
        self._normal_offsets, self._normal_starts = self._window_metadata(counts, self.motion_names)
        self._recovery, counts = self._load_clips(recovery_paths)
        self._recovery_offsets = self._recovery_starts = None
        if self.uses_recovery:
            self._recovery_offsets, self._recovery_starts = self._window_metadata(counts, self.recovery_motion_names)

        self.policy_history = torch.empty(num_steps_per_env, num_envs, self.obs_dim, device=self.device)
        self._history: torch.Tensor | None = None
        self._cursor = 0
        self.step = 0

    @staticmethod
    def _check_positive_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    def _load_clips(self, paths: list[Path]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Validate NPZs and align every raw field by name before concatenating frames."""
        keys = (
            "body_positions",
            "body_rotations",
            "body_linear_velocities",
            "body_angular_velocities",
            "dof_positions",
            "dof_velocities",
        )
        tensors = {name: [] for name in self.raw_shapes}
        counts = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                required = {*keys, "body_names", "dof_names", "root_link_name", "fps"}
                missing = required - set(data.files)
                if missing:
                    raise ValueError(f"Clip {path.name} is missing fields: {sorted(missing)}.")
                names = {}
                for label, expected in (("body_names", self.policy_body_names), ("dof_names", self.policy_joint_names)):
                    recorded = data[label]
                    if recorded.ndim != 1 or len(set(recorded.tolist())) != len(recorded):
                        raise ValueError(f"Clip {path.name} has invalid or duplicate {label}.")
                    absent = set(expected) - set(recorded.tolist())
                    if absent:
                        raise ValueError(f"Clip {path.name} is missing {label}: {sorted(absent)}.")
                    names[label] = recorded.tolist()
                root = data["root_link_name"]
                if root.size != 1 or str(root.reshape(-1)[0]) not in self.policy_body_names:
                    raise ValueError(f"Clip {path.name} must specify one known root_link_name.")
                root = str(root.reshape(-1)[0])
                if self._root_name is not None and root != self._root_name:
                    raise ValueError(f"Clip {path.name} root differs from {self._root_name!r}.")
                self._root_name = root
                fps = data["fps"]
                if fps.size != 1 or not np.isfinite(fps).all() or float(fps.reshape(-1)[0]) <= 0:
                    raise ValueError(f"Clip {path.name} fps must be finite and positive.")
                if not math.isclose(float(fps.reshape(-1)[0]) * self.step_dt, 1.0, rel_tol=1e-5):
                    raise ValueError(f"Clip {path.name} fps must match control frequency {1 / self.step_dt:g} Hz.")
                positions = data["body_positions"]
                if positions.ndim != 3 or positions.shape[0] < 1:
                    raise ValueError(f"Clip {path.name} body_positions must contain frames and bodies.")
                length = positions.shape[0]
                counts.append(length)
                for name, key in zip(self.raw_shapes, keys, strict=True):
                    body = name.startswith("body_")
                    recorded = names["body_names" if body else "dof_names"]
                    expected = self.policy_body_names if body else self.policy_joint_names
                    shape = (
                        (length, len(recorded), 4 if name == "body_quat_w" else 3) if body else (length, len(recorded))
                    )
                    values = data[key]
                    if values.shape != shape or not np.isfinite(values).all():
                        raise ValueError(f"Clip {path.name} field {key} must be finite with shape {shape}.")
                    if name == "body_quat_w" and (np.linalg.norm(values, axis=-1) < 1e-8).any():
                        raise ValueError(f"Clip {path.name} contains zero quaternions.")
                    indices = [recorded.index(n) for n in expected]
                    tensors[name].append(torch.tensor(values[:, indices], device=self.device, dtype=torch.float32))
        raw = {
            name: torch.cat(values) if values else torch.empty((0, *self.raw_shapes[name]), device=self.device)
            for name, values in tensors.items()
        }
        return raw, torch.tensor(counts, device=self.device, dtype=torch.long)

    def _window_metadata(self, counts: torch.Tensor, names: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        valid = counts >= self._span
        if not valid.any():
            raise ValueError(f"Expert pool {names!r} has no clip long enough; requires {self._span} frames.")
        return (counts.cumsum(0) - counts)[valid], (counts - self._span + 1)[valid]

    @property
    def uses_recovery(self) -> bool:
        """Whether recovery clips participate in sampling."""
        return bool(self.recovery_motion_names) and self.recovery_sample_ratio > 0

    @property
    def num_samples(self) -> int:
        """Number of policy windows in the current rollout."""
        return self.step * self.num_envs

    @property
    def policy_initialized(self) -> bool:
        """Whether an initial frame has seeded the raw policy history."""
        return self._history is not None

    def decode_raw(self, packed: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unpack raw state while preserving arbitrary leading sample/history dimensions."""
        if packed.shape[-1] != self.raw_dim:
            raise ValueError(f"Raw AMP observation width must be {self.raw_dim}, got {packed.shape[-1]}.")
        widths = [math.prod(shape) for shape in self.raw_shapes.values()]
        return {
            name: part.reshape(*packed.shape[:-1], *shape)
            for (name, shape), part in zip(self.raw_shapes.items(), packed.split(widths, dim=-1), strict=True)
        }

    def _policy_frame(self, raw_obs: torch.Tensor) -> torch.Tensor:
        if raw_obs.shape != (self.num_envs, self.raw_dim):
            raise ValueError(f"Raw policy observation must have shape ({self.num_envs}, {self.raw_dim}).")
        return raw_obs.detach().to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def initialize_policy(self, raw_obs: torch.Tensor) -> None:
        """Fill every history entry with its environment's initial raw frame."""
        if self.step:
            raise RuntimeError("Cannot reinitialize policy history during a collected rollout.")
        raw = self._policy_frame(raw_obs)
        self._history = raw[:, None].expand(-1, self._span, -1).clone()
        self._cursor = 0

    @torch.no_grad()
    def add_policy_step(
        self, raw_obs: torch.Tensor, dones: torch.Tensor, final_raw_obs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Record post-action windows, then seed done environments from their new reset frames.

        ``raw_obs`` contains Same-Step autoreset states. ``final_raw_obs`` is the full environment
        batch captured before reset; only done rows are used. Missing terminal data raises instead of
        training on the reset pose. Returns the terminal-corrected features used for AMP rewards.
        """
        if self._history is None:
            raise RuntimeError("Call initialize_policy before collecting policy steps.")
        if self.step >= self.num_steps_per_env:
            raise OverflowError("AMP rollout is full; call clear_rollout after training.")
        raw = self._policy_frame(raw_obs)
        if dones.numel() != self.num_envs:
            raise ValueError("dones must contain one flag per environment.")
        done = dones.reshape(-1).to(device=self.device, dtype=torch.bool)
        has_done = bool(done.any())
        terminal = raw
        if has_done:
            if final_raw_obs is None:
                raise ValueError("Done environments require final_raw_obs; enable compute_final_obs.")
            terminal = raw.clone()
            terminal[done] = self._policy_frame(final_raw_obs)[done]
        self._cursor = (self._cursor + 1) % self._span
        self._history[:, self._cursor] = terminal
        indices = (self._cursor - self._span + 1 + self._frame_steps) % self._span
        features = self.encoder.encode(self.decode_raw(self._history[:, indices])).flatten(1)
        self.policy_history[self.step].copy_(features)
        self._store_replay(features)
        self.step += 1
        if has_done:
            self._history[done] = raw[done, None, :]
        return features

    def _store_replay(self, features: torch.Tensor) -> None:
        """Append one step of policy features to the ring buffer, overwriting the oldest rows."""
        if self._replay is None:
            return
        count = features.shape[0]
        if count >= self.replay_buffer_size:
            # A single step already fills the pool; keep only the newest rows.
            self._replay.copy_(features[-self.replay_buffer_size :])
            self._replay_cursor = 0
            self._replay_filled = self.replay_buffer_size
            return
        end = self._replay_cursor + count
        if end <= self.replay_buffer_size:
            self._replay[self._replay_cursor : end].copy_(features)
        else:
            head = self.replay_buffer_size - self._replay_cursor
            self._replay[self._replay_cursor :].copy_(features[:head])
            self._replay[: end - self.replay_buffer_size].copy_(features[head:])
        self._replay_cursor = end % self.replay_buffer_size
        self._replay_filled = min(self.replay_buffer_size, max(end, self._replay_filled))

    @property
    def policy_sample_count(self) -> int:
        """Number of available policy windows, including retained replay samples."""
        return self.num_samples if self._replay is None else self._replay_filled

    def policy_samples(self) -> torch.Tensor:
        """Policy features available for discriminator training in this update.

        Returns the replay buffer contents when enabled, otherwise the current rollout.
        """
        if self._replay is None:
            if not self.num_samples:
                raise RuntimeError("Policy rollout is empty; collect policy steps before sampling.")
            return self.policy_history[: self.step].flatten(0, 1)
        if not self._replay_filled:
            raise RuntimeError("Policy replay buffer is empty; collect policy steps before sampling.")
        return self._replay[: self._replay_filled]

    @torch.no_grad()
    def sample_policy(self, batch_size: int) -> torch.Tensor:
        """Draw exactly ``batch_size`` policy windows with replacement from the active pool."""
        self._check_positive_int("batch_size", batch_size)
        policy = self.policy_samples()
        return policy[torch.randint(policy.shape[0], (batch_size,), device=self.device)]

    @torch.no_grad()
    def sample_raw_frames(self, num_samples: int) -> dict[str, torch.Tensor]:
        """Sample raw expert windows with a shared frame index across all six state fields."""
        self._check_positive_int("num_samples", num_samples)
        recovery_count = int(num_samples * self.recovery_sample_ratio) if self.uses_recovery else 0
        draws = []
        for count, raw, offsets, starts in (
            (num_samples - recovery_count, self._normal, self._normal_offsets, self._normal_starts),
            (recovery_count, self._recovery, self._recovery_offsets, self._recovery_starts),
        ):
            if count == 0:
                continue
            assert offsets is not None and starts is not None
            clips = torch.randint(offsets.numel(), (count,), device=self.device)
            first = (torch.rand(count, device=self.device) * starts[clips]).long()
            ids = offsets[clips, None] + first[:, None] + self._frame_steps
            draws.append({name: values[ids] for name, values in raw.items()})
        order = torch.randperm(num_samples, device=self.device)
        return {name: torch.cat([draw[name] for draw in draws])[order] for name in self.raw_shapes}

    @torch.no_grad()
    def sample_expert(self, batch_size: int) -> torch.Tensor:
        """Generate invariant expert inputs with the same encoder and layout as policy inputs."""
        return self.encoder.encode(self.sample_raw_frames(batch_size)).flatten(1)

    def mini_batch_generator(
        self, num_mini_batches: int, num_learning_epochs: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Yield equally sized policy/expert pairs for each PPO optimization step.

        Replay uses random sampling with replacement and PPO's flat mini-batch size. Without
        replay, shuffle all current-rollout samples once per epoch, retaining uneven final batches.
        Expert windows are drawn afresh for every pair through the shared feature encoder.
        """
        self._check_positive_int("num_mini_batches", num_mini_batches)
        self._check_positive_int("num_learning_epochs", num_learning_epochs)
        if not self.num_samples or num_mini_batches > self.num_samples:
            raise ValueError("AMP minibatch count must not exceed the nonempty rollout sample count.")
        if self._replay is not None:
            batch_size = self.num_samples // num_mini_batches
            for _ in range(num_learning_epochs * num_mini_batches):
                yield self.sample_policy(batch_size), self.sample_expert(batch_size)
            return
        policy = self.policy_samples()
        sample_count = policy.shape[0]
        for _ in range(num_learning_epochs):
            for ids in torch.tensor_split(torch.randperm(sample_count, device=self.device), num_mini_batches):
                yield policy[ids], self.sample_expert(ids.numel())

    def clear_rollout(self) -> None:
        """Clear rollout samples while retaining raw history, the ring cursor, and the replay buffer."""
        self.step = 0
