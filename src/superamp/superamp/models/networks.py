"""Differentiable dynamics and velocity actor networks."""

from __future__ import annotations

import torch
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict
from torch import nn

from superamp.representations import StateRepresentation


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


class VelocityActor(nn.Module):
    """Map proprioceptive history and a velocity goal to joint actions."""

    def __init__(
        self, obs_dim: int, goal_dim: int, action_dim: int, hidden_dims: tuple[int, ...], history_length: int
    ) -> None:
        super().__init__()
        self._aux_loss = torch.empty(0)
        self.obs_normalizer = _StableEmpiricalNormalization(obs_dim)
        self.goal_normalizer = _StableEmpiricalNormalization(goal_dim)
        self.network = build_mlp(history_length * obs_dim + goal_dim, hidden_dims, action_dim)

    def forward(self, observations: TensorDict) -> torch.Tensor:
        sensor = self.obs_normalizer(observations["actor"]).flatten(-2)
        goal = self.goal_normalizer(observations["goal"])
        action = self.network(torch.cat((sensor, goal), dim=-1))
        self._aux_loss = action.new_zeros(()).expand(action.shape[:-1])
        return action


    @torch.jit.unused
    @torch.no_grad()
    def update_normalization(self, observations: TensorDict) -> None:
        self.obs_normalizer.update(observations["actor"][:, -1])
        self.goal_normalizer.update(observations["goal"])
