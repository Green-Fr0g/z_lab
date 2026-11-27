"""Policy optimization through differentiable model rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict
from torch import nn

from superamp.envs.world_model_env import WorldModelEnv
from superamp.learners.cuda_graph import CapturedUpdate, graph_state
from superamp.learners.optimizer import build_optimizers, optimize_step


@dataclass
class BpttPolicyLearnerCfg:
    """Model rollout and actor optimization settings."""

    learning_rate: float = 2e-4
    num_batches: int = 1
    batch_size: int = 4096
    horizon: int = 32
    gamma: float = 1.0
    max_grad_norm: float = 1.0
    survival_masking: bool = True
    beta2: float = 0.95
    grad_clip_value: float = 0.1


@dataclass
class ActorCriticPolicyLearnerCfg(BpttPolicyLearnerCfg):
    """BPTT settings with critic optimization and GAE."""

    gamma: float = 0.99
    lam: float = 0.95
    critic_updates: int = 5
    critic_mini_batches: int = 4


def gae_returns(
    values: torch.Tensor,
    last_value: torch.Tensor,
    rewards: torch.Tensor,
    term_probabilities: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute GAE returns where soft termination discounts future values."""
    returns = torch.empty_like(values)
    advantage = torch.zeros_like(last_value)
    next_value = last_value
    for step in reversed(range(values.shape[0])):
        alive = 1.0 - term_probabilities[step]
        delta = rewards[step] + alive * gamma * next_value - values[step]
        advantage = delta + alive * gamma * lam * advantage
        returns[step] = advantage + values[step]
        next_value = values[step]
    return returns


class BpttPolicyLearner:
    """Update the actor by backpropagating through a model rollout."""

    def __init__(
        self,
        actor: nn.Module,
        world_model_env: WorldModelEnv,
        cfg: BpttPolicyLearnerCfg,
    ) -> None:
        if cfg.horizon <= 0 or cfg.num_batches <= 0 or cfg.batch_size <= 0:
            raise ValueError("horizon, num_batches and batch_size must be positive")
        self.actor = actor
        self.world_model_env = world_model_env
        self.cfg = cfg
        device = next(actor.parameters()).device
        self.actor_parameters = list(actor.parameters())
        self.actor_optimizers = build_optimizers([actor], cfg.learning_rate, device, betas=(0.9, cfg.beta2))
        self._rollout_actor = torch.compile(actor) if device.type == "cuda" else actor
        self._captured_update = CapturedUpdate(self._update_from, self._graph_state)
        for optimizer in self.actor_optimizers:
            optimizer.register_load_state_dict_post_hook(self._reset_captured_update)

    def update(self) -> dict[str, torch.Tensor]:
        """Run configured model rollouts and update the policy."""
        self.actor.train()
        self.world_model_env.dynamics.eval().requires_grad_(False)
        self.world_model_env.representation.eval().requires_grad_(False)
        try:
            starts = torch.stack([self.world_model_env.sample_initial_states() for _ in range(self.cfg.num_batches)])
            with self.world_model_env.discriminator.frozen():
                return self._captured_update(starts)
        finally:
            self.world_model_env.dynamics.requires_grad_(True)
            self.world_model_env.representation.requires_grad_(True)

    def _update_from(self, batches: TensorDict) -> dict[str, torch.Tensor]:
        logs: dict[str, list[torch.Tensor]] = {}
        for starts in batches.unbind(0):
            for name, value in self._train_batch(starts).items():
                logs.setdefault(name, []).append(value)
        scale = self.world_model_env.representation.dt * self.cfg.horizon
        reward = torch.stack(logs.pop("return")).mean() / scale
        metrics = {f"Policy/{name}": torch.stack(values).mean() for name, values in logs.items()}
        return {"Policy/reward": reward, **metrics}

    def _train_batch(self, starts: TensorDict) -> dict[str, torch.Tensor]:
        """Optimize the policy on one imagined rollout."""
        rollout, total_return, aux_loss, last_observations = self._collect_model_rollout(starts)
        actor_grad_norm = optimize_step(
            self.actor_optimizers,
            (aux_loss - total_return).mean(),
            self.actor_parameters,
            self.cfg.max_grad_norm,
            self.cfg.grad_clip_value,
        )
        critic_stats = self._fit_critic(rollout, last_observations)
        auxiliary_stats = getattr(self.actor, "auxiliary_stats", None)
        return {
            "return": total_return.mean().detach(),
            "actor_grad_norm": actor_grad_norm.detach(),
            **critic_stats,
            **(auxiliary_stats() if auxiliary_stats is not None else {}),
        }

    def _collect_model_rollout(
        self,
        starts: TensorDict,
    ) -> tuple[list[TensorDict], torch.Tensor, torch.Tensor, TensorDict]:
        observations = self.world_model_env.reset(starts)
        survival = torch.ones(self.cfg.batch_size, device=starts.device)
        total_return = torch.zeros_like(survival)
        aux_loss = torch.zeros_like(survival)
        auxiliary_loss = getattr(self.actor, "auxiliary_loss", None)
        rollout: list[TensorDict] = []
        for step in range(self.cfg.horizon):
            action = self._rollout_actor(observations)
            if auxiliary_loss is not None:
                aux_loss = aux_loss + auxiliary_loss()
            next_observations, reward, termination = self.world_model_env.step(action)
            if reward.device != total_return.device:
                raise RuntimeError(
                    "world-model rollout tensors are split across devices: "
                    f"return={total_return.device}, reward={reward.device}, "
                    f"action={action.device}, state={self.world_model_env.rep_state.device}, "
                    f"joint_limits={self.world_model_env.soft_joint_pos_limits.device}, "
                    f"sim_context={self.world_model_env.sim_ctx.device}"
                )
            if "critic" in observations:
                rollout.append(
                    TensorDict(
                        {
                            "critic": observations["critic"].detach(),
                            "reward": reward.detach(),
                            "termination": termination.detach(),
                        },
                        batch_size=[self.cfg.batch_size],
                    )
                )
            total_return = total_return + self.cfg.gamma**step * survival * reward
            if self.cfg.survival_masking:
                survival = survival * (1.0 - termination)
            observations = next_observations

        alive = survival if self.cfg.survival_masking else 1.0 - termination
        total_return = total_return + self._bootstrap(observations, alive)
        return rollout, total_return, aux_loss, observations

    def _bootstrap(self, observations: TensorDict, survival: torch.Tensor) -> torch.Tensor | float:
        """Return the value contribution at the rollout horizon."""
        return 0.0

    def _fit_critic(self, rollout: list[TensorDict], last_observations: TensorDict) -> dict[str, torch.Tensor]:
        """Return critic metrics for learners that train a value function."""
        return {}

    def _reset_captured_update(self, _optimizer: torch.optim.Optimizer) -> None:
        """Warm up a new graph after restoring optimizer tensor storage."""
        self._captured_update = CapturedUpdate(self._update_from, self._graph_state)

    def _graph_state(self) -> list[torch.Tensor]:
        env = self.world_model_env
        amp = env.discriminator
        return graph_state(
            (self.actor, env.dynamics, env.representation, amp.discriminator, amp.normalizer), self.actor_optimizers
        )


class ActorCriticPolicyLearner(BpttPolicyLearner):
    """Add value bootstrapping and GAE critic updates to BPTT."""

    cfg: ActorCriticPolicyLearnerCfg

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        world_model_env: WorldModelEnv,
        cfg: ActorCriticPolicyLearnerCfg,
    ) -> None:
        if cfg.critic_mini_batches <= 0 or cfg.batch_size * cfg.horizon < cfg.critic_mini_batches:
            raise ValueError("critic minibatch count exceeds the rollout sample count")
        if not 0.0 <= cfg.gamma < 1.0 or cfg.critic_updates < 1:
            raise ValueError("Actor-critic requires 0 <= gamma < 1 and at least one critic update")
        super().__init__(actor, world_model_env, cfg)

        self.critic = critic
        self.critic_parameters = list(critic.parameters())
        self.critic_optimizers = build_optimizers([critic], cfg.learning_rate, self.actor_parameters[0].device)
        for optimizer in self.critic_optimizers:
            optimizer.register_load_state_dict_post_hook(self._reset_captured_update)
        self._value_max = world_model_env.max_step_reward / (1.0 - cfg.gamma)

    def update(self) -> dict[str, torch.Tensor]:
        """Update both the actor and its value critic."""
        self.critic.train()
        return super().update()

    def _bootstrap(self, observations: TensorDict, survival: torch.Tensor) -> torch.Tensor:
        """Estimate the discounted final value while freezing critic weights."""
        self.critic.requires_grad_(False)
        try:
            value = self.critic(observations["critic"]).squeeze(-1).clamp(max=self._value_max)
        finally:
            self.critic.requires_grad_(True)
        return self.cfg.gamma**self.cfg.horizon * survival * value

    def _fit_critic(self, rollout: list[TensorDict], last_observations: TensorDict) -> dict[str, torch.Tensor]:
        steps = torch.stack(rollout)
        with torch.no_grad():
            values = self.critic(steps["critic"])
            last_value = self.critic(last_observations["critic"])
            returns = gae_returns(
                values,
                last_value,
                steps["reward"].unsqueeze(-1),
                steps["termination"].unsqueeze(-1),
                self.cfg.gamma,
                self.cfg.lam,
            ).clamp(max=self._value_max)
        critic_obs = steps["critic"].flatten(0, 1)
        target_values = returns.flatten(0, 1)
        sample_count = target_values.shape[0]
        usable_count = sample_count // self.cfg.critic_mini_batches * self.cfg.critic_mini_batches
        if usable_count == 0:
            raise ValueError("critic minibatches require at least one sample per minibatch")
        indices = torch.randperm(sample_count, device=target_values.device)[:usable_count]
        losses: list[torch.Tensor] = []
        for _ in range(self.cfg.critic_updates):
            for mini_batch in indices.chunk(self.cfg.critic_mini_batches):
                value_loss = (self.critic(critic_obs[mini_batch]) - target_values[mini_batch]).square().mean()
                optimize_step(
                    self.critic_optimizers,
                    value_loss,
                    self.critic_parameters,
                    self.cfg.max_grad_norm,
                )
                losses.append(value_loss.detach())
        return {"value_function": torch.stack(losses).mean()}

    def _graph_state(self) -> list[torch.Tensor]:
        return super()._graph_state() + graph_state((self.critic,), self.critic_optimizers)
