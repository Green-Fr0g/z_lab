# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Least-squares AMP discriminator over fixed-length observation windows."""

import torch
from torch import nn


class Discriminator(nn.Module):
    """Score a window of AMP features in chronological order."""

    def __init__(self, observation_dim: int, num_frames: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("The AMP discriminator requires at least one hidden layer")
        self.observation_dim = observation_dim
        self.num_frames = num_frames
        input_dim = observation_dim * num_frames
        layers: list[nn.Module] = []
        for output_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
            input_dim = output_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Return logits for features with trailing shape (num_frames, observation_dim)."""
        if windows.shape[-2:] != (self.num_frames, self.observation_dim):
            raise ValueError(f"Expected AMP windows with trailing shape ({self.num_frames}, {self.observation_dim})")
        return self.head(self.trunk(windows.flatten(-2)))

    def gradient_penalty(self, expert_windows: torch.Tensor, coefficient: float) -> torch.Tensor:
        """Penalize the input gradient of the logit over all expert frames."""
        inputs = expert_windows.detach().requires_grad_(True)
        gradients = torch.autograd.grad(self(inputs).sum(), inputs, create_graph=True)[0]
        return coefficient * gradients.flatten(-2).square().sum(dim=-1).mean()
