"""Supervised multi-step learning for world-model dynamics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from tensordict import TensorDict

from superamp.learners.cuda_graph import CapturedUpdate, graph_state
from superamp.learners.optimizer import build_optimizers, optimize_step
from superamp.models import Dynamics
from superamp.representations import StateRepresentation


@dataclass
class WorldModelLearnerCfg:
    """Optimization and state-channel loss configuration."""

    learning_rate: float = 2e-4
    num_batches: int = 10
    batch_size: int = 2048
    horizon: int = 8
    nominal_root_position_error: float = 0.01
    nominal_root_orientation_error: float = 0.02
    nominal_joint_position_error: float = 0.1
    nominal_joint_velocity_error: float = 3.14
    nominal_body_position_error: float = 0.05
    nominal_body_orientation_error: float = 0.1
    nominal_body_linear_velocity_error: float = 1.0
    nominal_body_angular_velocity_error: float = 3.14
    max_grad_norm: float = 1.0
    compile: bool = False


class WorldModelLearner:
    """Fit differentiable dynamics to contiguous replay sequences."""

    def __init__(
        self,
        dynamics: Dynamics,
        representation: StateRepresentation,
        cfg: WorldModelLearnerCfg,
    ) -> None:
        self.dynamics = dynamics
        self.representation = representation
        self.cfg = cfg
        if cfg.num_batches <= 0 or cfg.batch_size <= 0 or cfg.horizon < 1:
            raise ValueError("world-model batches, batch size and horizon must be positive")
        self._parameters = list(dynamics.parameters())
        device = self._parameters[0].device
        self.optimizers = build_optimizers(
            [dynamics],
            cfg.learning_rate,
            device,
        )
        self._loss_weights = {
            name: 1.0 / getattr(cfg, config_name) ** 2
            for name, config_name in representation.LOSS_WEIGHT_CONFIG.items()
        }
        self._represent = representation.represent
        self._predict_sequence = self._rollout
        self._state_loss = self._compute_losses
        self._captured_update = self._update_from
        if cfg.compile and device.type == "cuda":
            self._represent = torch.compile(self._represent)
            self._predict_sequence = torch.compile(self._predict_sequence)
            self._state_loss = torch.compile(self._state_loss)
            self._captured_update = CapturedUpdate(self._update_from, self._graph_state)

    @torch.no_grad()
    def update_normalization(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        """Update normalizers from newly collected states and processed actions."""
        rep_state = self.representation.represent(states.reshape(-1, states.shape[-1]))
        self.dynamics.state_normalizer.train()
        self.dynamics.action_normalizer.train()
        self.dynamics.state_normalizer.update(rep_state)
        self.dynamics.action_normalizer.update(self.dynamics.process_action(actions.reshape(-1, actions.shape[-1])))
        self.dynamics.state_normalizer.eval()
        self.dynamics.action_normalizer.eval()

    def update(self, sequences: Iterable[TensorDict]) -> dict[str, torch.Tensor]:
        """Apply one optimizer step per replay sequence batch."""
        self.dynamics.train()
        self.dynamics.requires_grad_(True)
        batches = torch.stack([sequence.select("state", "action", "next_state", "sim_ctx") for sequence in sequences])
        try:
            return self._captured_update(batches)
        finally:
            self.dynamics.eval()

    def _update_from(self, batches: TensorDict) -> dict[str, torch.Tensor]:
        totals: dict[str, list[torch.Tensor]] = {}
        for sequence in batches.unbind(0):
            initial = sequence["state"][:, 0]
            targets = sequence["next_state"]
            pred_rep_states, pred_aux_states = self._predict_sequence(
                self._represent(initial),
                sequence["action"],
                self.representation.extract_aux_state(initial),
                sequence["sim_ctx"][:, 0],
            )
            loss, group_losses = self._state_loss(
                pred_rep_states,
                pred_aux_states,
                self._represent(targets),
                self.representation.extract_aux_state(targets),
            )
            optimize_step(self.optimizers, loss, self._parameters, self.cfg.max_grad_norm)
            totals.setdefault("state", []).append(loss.detach())
            for name, value in group_losses.items():
                totals.setdefault(name, []).append(value.detach())
        return {f"Loss/{name}": torch.stack(values).mean() for name, values in totals.items()}

    def _rollout(
        self,
        rep_state: torch.Tensor,
        actions: torch.Tensor,
        aux_state: torch.Tensor,
        sim_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the states after every action with a recorded next state."""
        predictions_rep: list[torch.Tensor] = []
        predictions_aux: list[torch.Tensor] = []
        for step in range(actions.shape[1]):
            aux_state, rep_state = self.dynamics(rep_state, actions[:, step], aux_state, sim_context)
            predictions_rep.append(rep_state)
            predictions_aux.append(aux_state)
        return torch.stack(predictions_rep, dim=1), torch.stack(predictions_aux, dim=1)

    def _compute_losses(
        self,
        pred_rep_states: torch.Tensor,
        pred_aux_states: torch.Tensor,
        target_rep_states: torch.Tensor,
        target_aux_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted state loss and unweighted MSE per state group."""
        predicted_blocks = self.representation.loss_blocks(pred_rep_states, pred_aux_states)
        target_blocks = self.representation.loss_blocks(target_rep_states, target_aux_states)
        group_losses = {
            name: (predicted_blocks[name] - target_blocks[name]).square().mean() for name in predicted_blocks
        }
        widths = {name: value.shape[-1] for name, value in predicted_blocks.items()}
        denominator = sum(widths.values())
        total_loss = (
            sum(group_losses[name] * self._loss_weights[name] * widths[name] for name in group_losses) / denominator
        )
        return total_loss, group_losses

    def _graph_state(self) -> list[torch.Tensor]:
        return graph_state((self.dynamics, self.representation), self.optimizers)
