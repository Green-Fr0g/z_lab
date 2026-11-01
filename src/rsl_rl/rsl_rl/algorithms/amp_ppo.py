# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP-PPO implementation integrated with the RSL-RL 5.5 API."""

from __future__ import annotations

import math
from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.amp_discriminator import Discriminator
from rsl_rl.storage import AMPLoader, AMPReplayBuffer, RolloutStorage
from rsl_rl.utils import resolve_optimizer


class AMPPPO(PPO):
    """PPO with AMP reward shaping and jointly optimized motion discriminator."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        amp_motion_files: str,
        amp_body_names: tuple[str, ...],
        amp_anchor_name: str,
        amp_reward_coef: float = 0.1,
        amp_task_reward_lerp: float = 0.75,
        amp_replay_buffer_size: int = 100_000,
        amp_discr_hidden_dims: list[int] | None = None,
        amp_head_weight_decay: float = 0.1,
        amp_exclude_motion_names: tuple[str, ...] = (),
        amp_min_std: float = 0.05,
        **ppo_cfg: Any,
    ) -> None:
        """Initialize PPO and its expert-motion discriminator components."""
        super().__init__(actor, critic, storage, **ppo_cfg)
        if amp_task_reward_lerp < 0.0 or amp_task_reward_lerp > 1.0:
            raise ValueError("amp_task_reward_lerp must be in [0, 1]")

        self.amp_expert = AMPLoader(
            amp_motion_files,
            amp_body_names,
            amp_anchor_name,
            device=self.device,
            exclude_motion_names=amp_exclude_motion_names,
        )
        amp_obs = storage.observations["amp"]
        if amp_obs.shape[-1] != self.amp_expert.observation_dim:
            raise ValueError(
                "AMP observation dimension mismatch: environment provides "
                f"{amp_obs.shape[-1]}, motion data provides {self.amp_expert.observation_dim}"
            )
        self.discriminator = Discriminator(
            self.amp_expert.observation_dim, amp_discr_hidden_dims or [1024, 512, 256]
        ).to(self.device)
        self.amp_normalizer = EmpiricalNormalization(shape=[self.amp_expert.observation_dim], until=None).to(
            self.device
        )
        self.amp_reward_coef = amp_reward_coef
        self.amp_task_reward_lerp = amp_task_reward_lerp
        self.amp_min_std = amp_min_std
        self.amp_replay_buffer = AMPReplayBuffer(self.amp_expert.observation_dim, amp_replay_buffer_size, self.device)
        distribution = self.actor.distribution
        if distribution is not None and hasattr(distribution, "std_range"):
            distribution.std_range[0] = max(distribution.std_range[0], amp_min_std)
            distribution.log_std_range[0] = max(distribution.log_std_range[0], math.log(amp_min_std))
        self._amp_state: torch.Tensor | None = None
        self._amp_transition_count = 0

        optimizer_cls = resolve_optimizer(ppo_cfg.get("optimizer", "adam"))
        self.optimizer = optimizer_cls(
            [
                {"params": chain(self.actor.parameters(), self.critic.parameters()), "name": "policy"},
                {"params": self.discriminator.trunk.parameters(), "weight_decay": 1.0e-3, "name": "amp_trunk"},
                {
                    "params": self.discriminator.head.parameters(),
                    "weight_decay": amp_head_weight_decay,
                    "name": "amp_head",
                },
            ],
            lr=self.learning_rate,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and retain the current AMP features for the transition."""
        if "amp" not in obs:
            raise KeyError("AMP training requires an 'amp' observation group")
        self._amp_state = obs["amp"].detach().clone()
        return super().act(obs)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Add AMP rewards and transitions before recording the PPO rollout step."""
        if self._amp_state is None:
            raise RuntimeError("AMPPPO.act must be called before process_env_step")
        next_amp = obs["amp"].detach().clone()
        final_obs = extras.get("final_obs")
        if final_obs is not None and torch.any(dones):
            final_amp = final_obs["amp"] if isinstance(final_obs, (dict, TensorDict)) else None
            if final_amp is None:
                raise KeyError("extras['final_obs'] must contain the 'amp' observation group")
            done_mask = dones.bool().reshape(-1)
            next_amp[done_mask] = final_amp[done_mask]

        self.amp_replay_buffer.insert(self._amp_state, next_amp)
        amp_reward = self.discriminator.reward(
            self.amp_normalizer(self._amp_state), self.amp_normalizer(next_amp), self.amp_reward_coef
        )
        combined_rewards = (1.0 - self.amp_task_reward_lerp) * amp_reward + self.amp_task_reward_lerp * rewards
        self._amp_transition_count += self._amp_state.shape[0]
        self._amp_state = None
        super().process_env_step(obs, combined_rewards, dones, extras)

    def update(self) -> dict[str, float]:
        """Optimize PPO and AMP discriminator together on each minibatch."""
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise NotImplementedError("AMPPPO currently supports feed-forward actor and critic models")
        if self.rnd is not None or self.symmetry is not None:
            raise NotImplementedError("AMPPPO does not yet combine RND or symmetry losses")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_loss = 0.0
        mean_grad_penalty = 0.0
        mean_policy_logit = 0.0
        mean_expert_logit = 0.0
        updates = self.num_learning_epochs * self.num_mini_batches
        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        if mini_batch_size == 0:
            raise ValueError("PPO minibatch size must be positive")

        policy_batches = self.amp_replay_buffer.feed_forward_generator(updates, mini_batch_size)
        expert_batches = self.amp_expert.feed_forward_generator(updates, mini_batch_size)
        for batch, policy_pair, expert_pair in zip(
            self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs),
            policy_batches,
            expert_batches,
            strict=True,
        ):
            if self.normalize_advantage_per_mini_batch and batch.advantages.numel() > 1:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, self.actor.output_distribution_params
                    )
                    kl_mean = kl.mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr, src=0)
                        self.learning_rate = lr.item()
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))
            advantages = batch.advantages.squeeze(-1)
            surrogate_loss = torch.maximum(
                -advantages * ratio,
                -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param),
            ).mean()
            if self.use_clipped_value_loss:
                clipped_values = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum(
                    (values - batch.returns).square(), (clipped_values - batch.returns).square()
                ).mean()
            else:
                value_loss = (batch.returns - values).square().mean()

            policy_states, policy_next_states = policy_pair
            expert_states, expert_next_states = expert_pair
            with torch.no_grad():
                # Accumulate raw features from both distributions before normalizing either one.
                self.amp_normalizer.update(
                    torch.cat((policy_states, policy_next_states, expert_states, expert_next_states), dim=0)
                )
                policy_states = self.amp_normalizer(policy_states)
                policy_next_states = self.amp_normalizer(policy_next_states)
                expert_states = self.amp_normalizer(expert_states)
                expert_next_states = self.amp_normalizer(expert_next_states)
            policy_logits = self.discriminator(policy_states, policy_next_states)
            expert_logits = self.discriminator(expert_states, expert_next_states)
            amp_loss = 0.5 * ((expert_logits - 1.0).square().mean() + (policy_logits + 1.0).square().mean())
            grad_penalty = self.discriminator.gradient_penalty(expert_states, expert_next_states, coefficient=10.0)
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + amp_loss
                + grad_penalty
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._clamp_policy_std()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_penalty += grad_penalty.item()
            mean_policy_logit += policy_logits.mean().item()
            mean_expert_logit += expert_logits.mean().item()

        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        self.storage.clear()
        return {
            "value": mean_value_loss / updates,
            "surrogate": mean_surrogate_loss / updates,
            "entropy": mean_entropy / updates,
            "amp": mean_amp_loss / updates,
            "amp_grad_pen": mean_grad_penalty / updates,
            "amp_policy_logit": mean_policy_logit / updates,
            "amp_expert_logit": mean_expert_logit / updates,
            "amp_replay_size": float(self.amp_replay_buffer.size),
        }

    def _clamp_policy_std(self) -> None:
        """Keep the Gaussian action scale above the configured floor."""
        distribution = self.actor.distribution
        if distribution is not None:
            with torch.no_grad():
                if hasattr(distribution, "std_param"):
                    distribution.std_param.clamp_(min=self.amp_min_std)
                if hasattr(distribution, "log_std_param"):
                    distribution.log_std_param.clamp_(min=math.log(self.amp_min_std))

    def train_mode(self) -> None:
        """Switch policy, critic, and discriminator into training mode."""
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        """Switch policy, critic, and discriminator into evaluation mode."""
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        """Return PPO checkpoint data with AMP model and normalizer state."""
        saved = super().save()
        saved["amp_discriminator_state_dict"] = self.discriminator.state_dict()
        saved["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
        saved["amp_transition_count"] = self._amp_transition_count
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore PPO checkpoint data and AMP-specific state when available."""
        iteration = super().load(loaded_dict, load_cfg, strict)
        if loaded_dict.get("amp_discriminator_state_dict") is not None:
            self.discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
        if loaded_dict.get("amp_normalizer_state_dict") is not None:
            self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"], strict=strict)
        self._amp_transition_count = loaded_dict.get("amp_transition_count", 0)
        return iteration

    def broadcast_parameters(self) -> None:
        """Broadcast discriminator parameters alongside the base PPO models."""
        super().broadcast_parameters()
        for tensor in self.discriminator.state_dict().values():
            torch.distributed.broadcast(tensor, src=0)

    def reduce_parameters(self) -> None:
        """Synchronize policy, value, and discriminator gradients across workers."""
        params = chain(self.actor.parameters(), self.critic.parameters(), self.discriminator.parameters())
        from rsl_rl.utils import reduce_gradients_in_buckets

        reduce_gradients_in_buckets(params, self.gpu_world_size, self.grad_reduce_bucket_mb)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> AMPPPO:
        """Build actor, critic, storage, and AMP components from the native runner config."""
        if cfg["algorithm"].get("amp_motion_files") is None:
            raise KeyError("AMPPPO requires algorithm.amp_motion_files")
        return PPO.construct_algorithm(obs, env, cfg, device)
