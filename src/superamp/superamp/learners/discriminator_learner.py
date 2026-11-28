"""Least-squares AMP discriminator training and differentiable reward evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..models.discriminator import Discriminator
from ..models.networks import _StableEmpiricalNormalization

if TYPE_CHECKING:
    from ..storage import AMPReplayBuffer, ExpertTransitions


@dataclass
class DiscriminatorLearnerCfg:
    """AMP network, optimization, and reward settings."""

    hidden_dims: tuple[int, ...] = (1024, 512, 256)
    learning_rate: float = 1e-3
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    gradient_penalty: float = 10.0
    trunk_weight_decay: float = 1e-3
    head_weight_decay: float = 0.1
    reward_coef: float = 0.1
    task_coef: float = 0.75
    num_frames: int = 2
    frame_stride: int = 1


class DiscriminatorLearner:
    """Fit an AMP discriminator on expert and simulator observation windows."""

    def __init__(self, observation_dim: int, cfg: DiscriminatorLearnerCfg, device: str | torch.device = "cpu") -> None:
        if min(observation_dim, cfg.num_steps_per_env, cfg.num_learning_epochs, cfg.num_mini_batches) <= 0:
            raise ValueError("Observation width, batch size, and batch count must be positive")
        if not 0.0 <= cfg.task_coef <= 1.0:
            raise ValueError("task_coef must be in [0, 1]")
        if cfg.num_frames < 2 or cfg.frame_stride < 1:
            raise ValueError("AMP windows require num_frames >= 2 and frame_stride >= 1")
        self.cfg = cfg
        self.device = torch.device(device)
        self.discriminator = Discriminator(observation_dim, cfg.num_frames, cfg.hidden_dims).to(self.device)
        self.normalizer = _StableEmpiricalNormalization(shape=observation_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.discriminator.trunk.parameters(), "weight_decay": cfg.trunk_weight_decay},
                {"params": self.discriminator.head.parameters(), "weight_decay": cfg.head_weight_decay},
            ],
            lr=cfg.learning_rate,
        )

    def update(
        self, expert: ExpertTransitions, policy: AMPReplayBuffer, batch_size: int, num_batches: int
    ) -> dict[str, torch.Tensor]:
        """Update the discriminator and input statistics from expert and simulator batches."""
        width = self.discriminator.observation_dim
        if expert.observation_dim != width or policy.observation_dim != width:
            raise ValueError(f"Expert and simulator AMP features must both have width {width}")
        window = (self.discriminator.num_frames, self.cfg.frame_stride)
        if (expert.num_frames, expert.frame_stride) != window or (policy.num_frames, policy.frame_stride) != window:
            raise ValueError(f"Expert and simulator AMP windows must both use (num_frames, frame_stride)={window}")
        self.discriminator.train()
        self.normalizer.train()
        metrics = []
        for _ in range(num_batches):
            with torch.no_grad():
                expert_windows = expert.sample(batch_size).to(self.device)
                policy_windows = policy.sample(batch_size).to(self.device)
                self.normalizer.update(torch.cat((expert_windows.flatten(0, 1), policy_windows.flatten(0, 1))))
                expert_windows = self.normalizer(expert_windows)
                policy_windows = self.normalizer(policy_windows)

            expert_logits = self.discriminator(expert_windows)
            policy_logits = self.discriminator(policy_windows)
            classification_loss = 0.5 * ((expert_logits - 1.0).square().mean() + (policy_logits + 1.0).square().mean())
            gradient_penalty = self.discriminator.gradient_penalty(expert_windows, self.cfg.gradient_penalty)
            loss = classification_loss + gradient_penalty
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            metrics.append(
                torch.stack(
                    (
                        loss.detach(),
                        classification_loss.detach(),
                        gradient_penalty.detach(),
                        expert_logits.detach().mean(),
                        policy_logits.detach().mean(),
                    )
                )
            )

        loss, classification_loss, gradient_penalty, expert_logit, policy_logit = torch.stack(metrics).mean(dim=0)
        return {
            "AMP/loss": loss,
            "AMP/classification_loss": classification_loss,
            "AMP/gradient_penalty": gradient_penalty,
            "AMP/expert_logit": expert_logit,
            "AMP/policy_logit": policy_logit,
        }

    @contextmanager
    def frozen(self) -> Iterator[None]:
        """Hold discriminator parameters and normalization fixed during an actor rollout."""
        parameters = tuple(self.discriminator.parameters())
        requires_grad = tuple(parameter.requires_grad for parameter in parameters)
        discriminator_training = self.discriminator.training
        normalizer_training = self.normalizer.training
        self.discriminator.eval().requires_grad_(False)
        self.normalizer.eval()
        try:
            yield
        finally:
            self.discriminator.train(discriminator_training)
            self.normalizer.train(normalizer_training)
            for parameter, trainable in zip(parameters, requires_grad, strict=True):
                parameter.requires_grad_(trainable)

    def reward(self, windows: torch.Tensor) -> torch.Tensor:
        """Score chronological AMP windows with input gradients enabled."""
        logits = self.discriminator(self.normalizer(windows)).squeeze(-1)
        return self.cfg.reward_coef * (1.0 - 0.25 * (logits - 1.0).square()).clamp_min(0.0)
