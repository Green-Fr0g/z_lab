"""Per-environment replay storage and episode-aware transition sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass(frozen=True)
class FieldSpec:
    """Trailing field shape and dtype for replay storage."""

    dim: int = 0
    dtype: torch.dtype = torch.float32

    def shape(self, num_envs: int, buffer_size: int) -> tuple[int, ...]:
        return (num_envs, buffer_size, self.dim) if self.dim else (num_envs, buffer_size)


class ReplayBuffer:
    """Store synchronized per-environment transitions in a circular buffer."""

    def __init__(
        self,
        schema: dict[str, FieldSpec],
        buffer_size: int,
        num_envs: int,
        device: str | torch.device = "cpu",
    ) -> None:
        if not {"terminated", "truncated"} <= schema.keys():
            raise ValueError("schema needs terminated and truncated fields")
        if buffer_size <= 0 or num_envs <= 0:
            raise ValueError("buffer_size and num_envs must be positive")
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.num_transitions = 0
        self._step = 0
        self._storage = TensorDict(
            {
                name: torch.zeros(spec.shape(num_envs, buffer_size), dtype=spec.dtype, device=self.device)
                for name, spec in schema.items()
            },
            batch_size=[num_envs, buffer_size],
            device=self.device,
        )

    def insert(self, transitions: TensorDict) -> None:
        if transitions.batch_size != torch.Size([self.num_envs]):
            raise ValueError(f"expected batch_size=[{self.num_envs}], got {list(transitions.batch_size)}")
        if set(transitions.keys()) != set(self._storage.keys()):
            raise ValueError("transition fields do not match replay schema")
        self._storage[:, self._step] = transitions.to(self.device)
        self._step = (self._step + 1) % self.buffer_size
        self.num_transitions = min(self.num_transitions + 1, self.buffer_size)

    def sample_causal(self, sequence_length: int, batch_size: int) -> tuple[TensorDict, torch.Tensor]:
        """Sample history windows ending at uniformly sampled stored transitions.

        Return the windows and a boolean padding mask, both with batch shape
        ``(batch_size, sequence_length)``. Repeat the earliest available frame
        in the episode when the history extends past an episode boundary or
        the oldest stored transition.
        """
        if sequence_length <= 0 or sequence_length > self.buffer_size:
            raise ValueError("sequence_length must be in [1, buffer_size]")
        filled = self.num_transitions
        if filled == 0:
            raise RuntimeError("cannot sample causal history from an empty replay buffer")

        base = self._step if filled == self.buffer_size else 0
        env_indices = torch.randint(self.num_envs, (batch_size, 1), device=self.device)
        current_ages = torch.randint(filled, (batch_size, 1), device=self.device)
        offsets = torch.arange(1 - sequence_length, 1, device=self.device)
        ages = current_ages + offsets
        slots = (base + ages.clamp_min(0)) % self.buffer_size
        ended = self._storage["terminated"][env_indices, slots] | self._storage["truncated"][env_indices, slots]

        # Only an earlier transition can end the episode before the sampled frame.
        valid = (ages >= 0) & ~ended.bool()
        valid[:, -1] = True
        reachable = valid.flip(-1).int().cummin(dim=-1).values.flip(-1).bool()
        first_ages = torch.where(reachable, ages, current_ages).amin(dim=-1, keepdim=True)
        slots = (base + torch.where(reachable, ages, first_ages)) % self.buffer_size
        return self._storage[env_indices, slots], ~reachable

    def sample(self, sequence_length: int, batch_size: int) -> TensorDict:
        """Sample contiguous windows that do not cross a terminal transition."""
        if sequence_length <= 0 or sequence_length > self.buffer_size:
            raise ValueError("sequence_length must be in [1, buffer_size]")
        if self.num_transitions < sequence_length:
            raise RuntimeError("replay buffer does not contain a full sequence")
        filled = self.num_transitions
        base = self._step if filled == self.buffer_size else 0
        ages = torch.arange(self.buffer_size, device=self.device)
        ended = (self._storage["terminated"] | self._storage["truncated"])[:, (base + ages) % self.buffer_size].long()
        cumulative_ends = torch.cat(
            (
                ended.new_zeros(self.num_envs, 1),
                ended.cumsum(1),
                ended.new_zeros(self.num_envs, sequence_length - 1),
            ),
            dim=1,
        )
        valid = (
            cumulative_ends[:, sequence_length - 1 : sequence_length - 1 + self.buffer_size]
            == cumulative_ends[:, : self.buffer_size]
        )
        valid &= ages + sequence_length <= filled
        valid_counts = valid.sum(dim=1)
        if not (valid_counts > 0).any():
            raise RuntimeError(f"no episode contains a sequence of length {sequence_length}")
        env_indices = torch.multinomial((valid_counts > 0).float(), batch_size, replacement=True)
        choice = (torch.rand(batch_size, device=self.device) * valid_counts[env_indices]).long()
        selected_valid = valid[env_indices]
        start_ages = torch.searchsorted(selected_valid.long().cumsum(1), choice[:, None], right=True).squeeze(-1)
        slots = (base + start_ages[:, None] + torch.arange(sequence_length, device=self.device)) % self.buffer_size
        return self._storage[env_indices[:, None], slots]
