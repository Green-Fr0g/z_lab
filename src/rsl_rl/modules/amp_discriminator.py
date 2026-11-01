# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP discriminator and reward calculation."""

from __future__ import annotations

import torch
from torch import autograd, nn


class Discriminator(nn.Module):
    """Least-squares discriminator over consecutive AMP feature vectors."""

    def __init__(self, observation_dim: int, hidden_dims: list[int] | tuple[int, ...]) -> None:
        """Create the discriminator MLP for the configured AMP feature size."""
        super().__init__()
        if not hidden_dims:
            raise ValueError("The AMP discriminator requires at least one hidden layer")
        self.observation_dim = observation_dim
        layers: list[nn.Module] = []
        input_dim = observation_dim * 2
        for output_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
            input_dim = output_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, 1)

    def forward(self, states: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """Return discriminator logits for consecutive AMP states."""
        return self.head(self.trunk(torch.cat((states, next_states), dim=-1)))

    def reward(self, states: torch.Tensor, next_states: torch.Tensor, coefficient: float) -> torch.Tensor:
        """Compute the positive-label least-squares AMP reward."""
        with torch.no_grad():
            logits = self(states, next_states)
            return coefficient * torch.clamp(1.0 - 0.25 * (logits - 1.0).square(), min=0.0).squeeze(-1)

    def gradient_penalty(
        self, expert_states: torch.Tensor, expert_next_states: torch.Tensor, coefficient: float = 10.0
    ) -> torch.Tensor:
        """Penalize discriminator gradient magnitude on expert transitions."""
        inputs = torch.cat((expert_states, expert_next_states), dim=-1).detach().requires_grad_(True)
        logits = self.head(self.trunk(inputs))
        gradients = autograd.grad(
            outputs=logits,
            inputs=inputs,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
        )[0]
        return coefficient * gradients.square().sum(dim=-1).mean()
