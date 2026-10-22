# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules import MLP, EmpiricalNormalization


class AMPDiscriminator(nn.Module):
    """Score prepared motion-history tensors using a configured input dimension.

    The caller selects observations, assembles histories, and determines ``input_dim``.
    The discriminator consumes those tensors directly without interpreting observation
    keys or individual features. ``frame_dim`` lets every temporal position share normalization.

    The module mirrors AMP_mjlab's ``Discriminator``: a ReLU trunk of ``nn.Linear`` layers
    followed by a single linear head and the same quadratic style reward. Normalization uses
    RSL-RL's EmpiricalNormalization, shared by all frames and both data sources.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 512, 256),
        activation: str = "relu",
        obs_normalization: bool = True,
        amp_reward_coef: float = 1.0,
        task_reward_lerp: float = 0.0,
        obs_eps: float = 1e-2,
        obs_clip: float | None = 10.0,
        frame_dim: int | None = None,
    ) -> None:
        """Build the discriminator for prepared input tensors.

        Args:
            input_dim: Total number of input features, including any history frames.
                Determined by the caller before constructing the discriminator.
            hidden_dims: Hidden layer widths of the discriminator MLP.
            activation: Activation function used between hidden layers.
            obs_normalization: Whether to normalize histories with shared statistics.
            amp_reward_coef: Scale applied to the imitation reward before mixing.
            task_reward_lerp: Task reward fraction in ``[0, 1]``; zero uses only
                the imitation reward.
            obs_eps: Stabilizer added to the running standard deviation, using RSL-RL's
                ``EmpiricalNormalization`` default.
            obs_clip: Symmetric clamp applied to normalized histories, matching
                AMP_mjlab's ``clip_obs=10.0``. ``None`` disables it.
            frame_dim: Single-frame feature width. All frames share running statistics.
                Defaults to ``input_dim`` for a single flattened observation.
        """
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim!r}.")
        if not hidden_dims:
            raise ValueError("The AMP discriminator requires at least one hidden layer.")
        if not 0.0 <= task_reward_lerp <= 1.0:
            raise ValueError("task_reward_lerp must be in [0, 1].")
        if obs_eps <= 0 or (obs_clip is not None and obs_clip <= 0):
            raise ValueError("obs_eps must be positive and obs_clip must be positive or None.")

        self.input_dim = input_dim
        self.frame_dim = input_dim if frame_dim is None else frame_dim
        if (
            isinstance(self.frame_dim, bool)
            or not isinstance(self.frame_dim, int)
            or self.frame_dim < 1
            or input_dim % self.frame_dim
        ):
            raise ValueError("frame_dim must be a positive integer dividing input_dim.")
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp
        self.obs_normalization = obs_normalization
        self.obs_clip = obs_clip
        self.obs_normalizer = (
            EmpiricalNormalization(self.frame_dim, eps=obs_eps) if obs_normalization else nn.Identity()
        )
        self.mlp = MLP(self.input_dim, 1, hidden_dims, activation)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``(N, 1)`` for complete flattened histories.

        The caller must assemble the history before passing it to the discriminator.
        Normalization statistics are updated only by :meth:`update_normalization`.
        """
        self._validate_history(history)
        frames = history.reshape(-1, self.frame_dim)
        normalized = self.obs_normalizer(frames)
        if self.obs_normalization and self.obs_clip is not None:
            normalized = normalized.clamp(-self.obs_clip, self.obs_clip)
        normalized = normalized.reshape_as(history)
        return self.mlp(normalized)

    @torch.no_grad()
    def update_normalization(self, policy_history: torch.Tensor, expert_history: torch.Tensor) -> None:
        """Update shared statistics from raw policy and expert history batches.

        Both inputs have shape ``(N, input_dim)``. All temporal positions contribute to the same statistics;
        equally sized pairs give both sources equal weight. Evaluation mode freezes statistics.
        The built-in normalizer synchronizes statistics when distributed training is initialized.
        """
        self._validate_history(policy_history)
        self._validate_history(expert_history)
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.update(torch.cat([policy_history, expert_history], dim=0).reshape(-1, self.frame_dim))

    def compute_gradient_penalty(self, expert_history: torch.Tensor, lambda_: float = 10.0) -> torch.Tensor:
        """Penalize squared input-gradient norms on raw expert histories.

        Following AMP_mjlab, the penalty bypasses normalization and regularizes the MLP in raw
        feature space. Running statistics and input clipping therefore do not affect this term.

        Args:
            expert_history: Raw expert histories shaped ``(N, input_dim)``.
            lambda_: Weight of the zero-centered gradient penalty.

        Returns:
            A scalar loss differentiable with respect to discriminator parameters.
            The input tensor and its gradient state are left untouched.
        """
        self._validate_history(expert_history)
        expert_input = expert_history.detach().clone().requires_grad_(True)
        logits = self.mlp(expert_input)
        gradient = torch.autograd.grad(logits.sum(), expert_input, create_graph=True)[0]
        return lambda_ * gradient.square().sum(dim=-1).mean()

    def compute_loss(
        self, policy_history: torch.Tensor, expert_history: torch.Tensor, grad_pen_coef: float = 10.0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the least-squares AMP objective and detached scalar training metrics.

        Both batches contain raw, encoded histories of shape ``(N, input_dim)``. Predictions
        use current normalization statistics, while the gradient penalty uses raw features.
        Statistics are updated separately after the joint PPO/discriminator optimizer step.
        """
        policy_pred = self(policy_history)
        expert_pred = self(expert_history)
        policy_loss = F.mse_loss(policy_pred, -torch.ones_like(policy_pred))
        expert_loss = F.mse_loss(expert_pred, torch.ones_like(expert_pred))
        amp_loss = 0.5 * (expert_loss + policy_loss)
        grad_pen = (
            self.compute_gradient_penalty(expert_history, lambda_=grad_pen_coef)
            if grad_pen_coef > 0
            else amp_loss.new_zeros(())
        )
        metrics = {
            "amp_loss": amp_loss.detach(),
            "amp_expert_loss": expert_loss.detach(),
            "amp_policy_loss": policy_loss.detach(),
            "amp_grad_pen": grad_pen.detach(),
            "amp_policy_pred": policy_pred.detach().mean(),
            "amp_expert_pred": expert_pred.detach().mean(),
        }
        return amp_loss + grad_pen, metrics

    @torch.no_grad()
    def reward(
        self, policy_history: torch.Tensor, task_reward: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute quadratic AMP rewards and optionally mix task rewards.

        Args:
            policy_history: Raw policy histories shaped ``(N, input_dim)``.
            task_reward: Task rewards shaped ``(N,)`` or ``(N, 1)``, required when
                ``task_reward_lerp`` is positive.

        Returns:
            Mixed rewards and discriminator logits, both shaped ``(N,)``.
            The imitation reward is ``coef * max(0, 1 - 0.25 * (logit - 1)^2)``.
        """
        if self.task_reward_lerp > 0:
            if task_reward is None:
                raise ValueError("task_reward is required when task_reward_lerp > 0.")
            if task_reward.shape not in ((policy_history.shape[0],), (policy_history.shape[0], 1)):
                raise ValueError("task_reward must have shape (N,) or (N, 1) matching the policy batch.")
        logits = self(policy_history).squeeze(-1)
        reward = self.amp_reward_coef * (1.0 - 0.25 * (logits - 1.0).square()).clamp(min=0.0)
        if self.task_reward_lerp > 0 and task_reward is not None:
            reward = (1.0 - self.task_reward_lerp) * reward + self.task_reward_lerp * task_reward.reshape(-1)
        return reward, logits

    def _validate_history(self, history: torch.Tensor) -> None:
        """Reject tensors whose feature dimension differs from the configured input dimension."""
        if history.ndim != 2 or history.shape[-1] != self.input_dim:
            raise ValueError(f"Expected flattened AMP histories of shape (N, {self.input_dim}), got {history.shape}.")

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Convert legacy AMP moments before loading the built-in normalizer's buffers."""
        normalizer_prefix = f"{prefix}obs_normalizer."
        mean = state_dict.pop(f"{normalizer_prefix}mean", None)
        if mean is not None:
            means = mean.reshape(-1, self.frame_dim)
            variances = state_dict.pop(f"{normalizer_prefix}var").reshape_as(means)
            pooled_mean = means.mean(dim=0, keepdim=True)
            pooled_var = (variances + (means - pooled_mean).square()).mean(dim=0, keepdim=True)
            state_dict[f"{normalizer_prefix}_mean"] = pooled_mean
            state_dict[f"{normalizer_prefix}_var"] = pooled_var
            state_dict[f"{normalizer_prefix}_std"] = pooled_var.sqrt()
            count_key = f"{normalizer_prefix}count"
            state_dict[count_key] = (state_dict[count_key] * means.shape[0]).round().long()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
