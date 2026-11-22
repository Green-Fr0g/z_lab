"""Differentiable dynamics and multi-task actor networks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict
from torch import nn

from supermimic.representations import StateRepresentation


class _StableEmpiricalNormalization(EmpiricalNormalization):
    """Keep the running standard-deviation buffer at a stable memory address."""

    @torch.jit.unused
    def update(self, values: torch.Tensor) -> None:
        standard_deviation = self._std
        super().update(values)
        standard_deviation.copy_(self._std)
        self._std = standard_deviation


def build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    """Build a Linear, LayerNorm and GELU MLP."""
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class Dynamics(nn.Module):
    """Predict one state transition for a configured representation."""

    def __init__(
        self,
        representation: StateRepresentation,
        num_actions: int,
        sim_context_dim: int,
        action_scale: torch.Tensor,
        action_offset: torch.Tensor,
        hidden_dims: tuple[int, ...] = (2048, 2048, 2048),
    ) -> None:
        super().__init__()
        if action_scale.shape != (num_actions,) or action_offset.shape != (num_actions,):
            raise ValueError("action scale and offset must match num_actions")
        self.representation = representation
        self.num_actions = num_actions
        self.register_buffer("action_scale", action_scale.clone())
        self.register_buffer("action_offset", action_offset.clone())
        self.state_normalizer = _StableEmpiricalNormalization(representation.rep_state_dim)
        self.action_normalizer = _StableEmpiricalNormalization(num_actions)
        self.network = build_mlp(
            representation.rep_state_dim + num_actions + sim_context_dim,
            hidden_dims,
            representation.prediction_dim,
        )

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        """Apply the task action term's scale and offset."""
        return action * self.action_scale + self.action_offset

    def forward(
        self,
        rep_state: torch.Tensor,
        action: torch.Tensor,
        aux_state: torch.Tensor,
        sim_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict auxiliary root state and generalized state at the next step."""
        state = self.state_normalizer(rep_state)
        processed_action = self.action_normalizer(self.process_action(action))
        prediction = self.network(torch.cat((state, processed_action, sim_context), dim=-1))
        return self.representation.apply_prediction(aux_state, rep_state, prediction)

    def normalizers(self) -> tuple[_StableEmpiricalNormalization, _StableEmpiricalNormalization]:
        return self.state_normalizer, self.action_normalizer


@dataclass(frozen=True)
class TaskSpec:
    """Actor input contract for a task encoder."""

    name: str
    goal_dim: int


class MultiEncoderAE(nn.Module):
    """Task-specific goal encoders and a shared latent-conditioned decoder.

    Each encoder reads a contiguous slice of its task goal from the batch. The
    decoder receives the bounded latent and the flattened observation history.
    During training, latent BC can align the other encoders with the first task
    encoder on the same observations.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        task_specs: list[TaskSpec],
        slot_mix: dict[str, float],
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256),
        history_length: int = 1,
        latent_bc_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if not task_specs:
            raise ValueError("task_specs must contain at least one task")
        self.obs_dim = obs_dim
        self.history_length = history_length
        self.latent_dim = latent_dim
        self.latent_bc_weight = latent_bc_weight
        self.task_specs = task_specs
        self.task_names = [task.name for task in task_specs]
        self._anchor = self.task_names[0]
        self._aux_loss = torch.empty(0)
        self.obs_normalizer = _StableEmpiricalNormalization(obs_dim)
        self.goal_normalizers = nn.ModuleDict(
            {task.name: _StableEmpiricalNormalization(task.goal_dim) for task in task_specs}
        )
        self.encoders = nn.ModuleDict(
            {task.name: build_mlp(task.goal_dim, hidden_dims, latent_dim) for task in task_specs}
        )
        self.decoder = build_mlp(latent_dim + history_length * obs_dim, hidden_dims, action_dim)
        self.set_slot_mix(slot_mix)

    def set_slot_mix(self, slot_mix: dict[str, float]) -> None:
        unknown = set(slot_mix) - set(self.task_names)
        if unknown:
            raise ValueError(f"slot_mix contains unknown tasks {sorted(unknown)}")
        fractions = [slot_mix.get(name, 0.0) for name in self.task_names]
        if any(value < 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError("slot_mix values must be non-negative and sum to one")
        self._slot_fractions = fractions

    def slot_ranges(self, batch_size: int) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = 0
        for index, fraction in enumerate(self._slot_fractions):
            stop = batch_size if index == len(self._slot_fractions) - 1 else start + round(batch_size * fraction)
            ranges.append((start, stop))
            start = stop
        return ranges

    def forward(self, observations: TensorDict) -> torch.Tensor:
        slots = self.slot_ranges(observations.shape[0])
        encoded: list[torch.Tensor] = []
        for name, (start, stop) in zip(self.task_names, slots, strict=True):
            if start == stop:
                continue
            goal = self.goal_normalizers[name](observations[f"goal_{name}"][start:stop])
            encoded.append(torch.tanh(self.encoders[name](goal)))
        self._aux_loss = self._compute_aux_loss(observations, slots)
        proprioception = self.obs_normalizer(observations["history"])
        return self.decoder(torch.cat((torch.cat(encoded), proprioception.flatten(start_dim=1)), dim=-1))

    def _compute_aux_loss(self, observations: TensorDict, slots: list[tuple[int, int]]) -> torch.Tensor:
        """Compute per-sample latent BC with gradients only for non-anchor encoders."""
        loss = observations["history"].new_zeros(observations.shape[0])
        if not self.training or self.latent_bc_weight == 0.0:
            return loss

        anchor = self._anchor
        for name, (start, stop) in zip(self.task_names, slots, strict=True):
            if name == anchor or start == stop:
                continue
            goal = self.goal_normalizers[name](observations[f"goal_{name}"][start:stop].detach())
            latent = torch.tanh(self.encoders[name](goal))
            with torch.no_grad():
                anchor_goal = self.goal_normalizers[anchor](observations[f"goal_{anchor}"][start:stop])
                anchor_latent = torch.tanh(self.encoders[anchor](anchor_goal))
            loss[start:stop] = self.latent_bc_weight * (latent - anchor_latent).square().mean(dim=-1)
        return loss

    def auxiliary_loss(self) -> torch.Tensor:
        """Return the per-sample auxiliary loss from the latest forward pass."""
        return self._aux_loss

    def auxiliary_stats(self) -> dict[str, torch.Tensor]:
        """Return the mean latent BC loss for logging."""
        if self._aux_loss.numel() == 0:
            return {}
        return {"bc_loss": self._aux_loss.detach().mean()}

    @torch.jit.unused
    def update_normalization(self, observations: TensorDict) -> None:
        self.obs_normalizer.update(observations["history"][:, -1])
        for name, (start, stop) in zip(self.task_names, self.slot_ranges(observations.shape[0]), strict=True):
            if start != stop:
                self.goal_normalizers[name].update(observations[f"goal_{name}"][start:stop])
