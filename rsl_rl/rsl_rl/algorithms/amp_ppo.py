# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import warnings
from collections.abc import Generator, Sequence
from itertools import chain
from typing import Any, cast

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import AMPDiscriminator, MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import AmpDataset, reduce_gradients_in_buckets, resolve_class

from .ppo import PPO


class AMPPPO(PPO):
    """PPO with shared expert/policy feature generation owned by AmpDataset.

    The environment provides raw single-frame observations. Actor/critic observations and PPO
    optimization retain the standard model interfaces; AMP history and rollout data live in the
    dataset, and discriminator width follows its encoded feature layout.
    """

    amp_dataset: AmpDataset
    """Expert clips, policy history, rollout samples, and replay."""

    discriminator: AMPDiscriminator
    """Motion discriminator trained jointly with the actor and critic."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        amp_cfg: dict | None = None,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO and construct AMP components from the resolved extension configuration.

        ``amp_cfg.dataset_cfg`` provides motion paths, robot metadata, features, and temporal
        sampling. :meth:`construct_algorithm` resolves the environment metadata before construction.
        ``discriminator_cfg`` follows RSL-RL's ``class_name`` model configuration convention.
        The extension also configures gradient penalty, weight decay, and an optional action-noise
        floor. Epochs, mini-batches, optimizer, and learning-rate schedule are inherited from PPO.
        Only actor/critic gradients are clipped, as in AMP_mjlab.
        """
        super().__init__(actor, critic, storage, device=device, multi_gpu_cfg=multi_gpu_cfg, **kwargs)
        config = dict(amp_cfg or {})
        self.amp_policy_key = config.pop("amp_policy_key", "amp_policy")
        dataset_cfg = config.pop("dataset_cfg", None)
        legacy_dataset = config.pop("amp_dataset", None)
        if legacy_dataset is not None:
            warnings.warn(
                "amp_cfg['amp_dataset'] is deprecated. Pass constructor arguments through amp_cfg['dataset_cfg'].",
                DeprecationWarning,
                stacklevel=2,
            )
            if dataset_cfg is not None:
                raise ValueError("Provide dataset_cfg or the deprecated amp_dataset, not both.")
            self.amp_dataset = legacy_dataset
        else:
            if dataset_cfg is None:
                raise ValueError("AMP requires dataset_cfg with motion paths, robot metadata, and step_dt.")
            self.amp_dataset = AmpDataset(
                **dataset_cfg,
                num_envs=storage.num_envs,
                num_steps_per_env=storage.num_transitions_per_env,
                device=self.device,
            )
        if (
            self.amp_dataset.num_envs != storage.num_envs
            or self.amp_dataset.num_steps_per_env != storage.num_transitions_per_env
        ):
            raise ValueError("AMP dataset and PPO storage must have the same rollout capacity.")
        discriminator_class, discriminator_cfg = resolve_class(
            {"class_name": "rsl_rl.models:AMPDiscriminator", **config.pop("discriminator_cfg", {})}
        )
        self.discriminator = discriminator_class(
            self.amp_dataset.obs_dim, frame_dim=self.amp_dataset.frame_dim, **discriminator_cfg
        ).to(device)
        for name, value in (
            ("num_learning_epochs", self.num_learning_epochs),
            ("num_mini_batches", self.num_mini_batches),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"AMP {name} must be a positive integer, got {value!r}.")
        if self.num_mini_batches > storage.num_envs * storage.num_transitions_per_env:
            raise ValueError("AMP num_mini_batches must not exceed the rollout sample count.")

        self.grad_pen_coef = config.pop("grad_pen_coef", 10.0)
        # AMP_mjlab keeps the action noise above a per-joint floor so the Gaussian never collapses.
        self.min_normalized_std = self._resolve_min_std(config.pop("min_normalized_std", None))
        trunk_weight_decay = config.pop("trunk_weight_decay", 1e-3)
        head_weight_decay = config.pop("head_weight_decay", 1e-1)
        if config:
            raise ValueError(f"Unknown AMP configuration keys: {sorted(config)}.")
        if not math.isfinite(self.grad_pen_coef) or self.grad_pen_coef < 0 or self.max_grad_norm <= 0:
            raise ValueError("AMP grad_pen_coef must be nonnegative and max_grad_norm must be positive.")
        self._configure_discriminator_optimizer(trunk_weight_decay, head_weight_decay)

        self._amp_batches: Generator[tuple[torch.Tensor, torch.Tensor], None, None] | None = None
        self._amp_normalization_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        self._amp_totals: dict[str, torch.Tensor] = {}
        self._amp_updates = 0

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Initialize raw history once and pass only PPO observations into its transition."""
        if self.transition.actions is not None:
            raise RuntimeError("process_env_step must follow each AMP act call.")
        if not self.amp_dataset.policy_initialized:
            self.amp_dataset.initialize_policy(obs[self.amp_policy_key])
        return super().act(obs.select(*self.storage.observations.keys()))

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> torch.Tensor:
        """Store terminal-corrected features and return mixed task/style rewards for logging."""
        if self.transition.actions is None:
            raise RuntimeError("AMPPPO.process_env_step must follow act.")
        final_obs = extras.get("final_obs")
        policy_history = self.amp_dataset.add_policy_step(
            obs[self.amp_policy_key],
            dones,
            None if final_obs is None else final_obs[self.amp_policy_key],
        )
        amp_rewards, _ = self.discriminator.reward(policy_history, rewards)
        super().process_env_step(obs.select(*self.storage.observations.keys()), amp_rewards, dones, extras)
        return amp_rewards

    def update(self) -> dict[str, float]:
        """Update AC and the discriminator together on each mini-batch, then clear the rollout."""
        if self.amp_dataset.step != self.storage.step or self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("PPO and AMP storage must contain the same complete rollout before update().")
        self._amp_totals.clear()
        self._amp_updates = 0
        self._amp_batches = self.amp_dataset.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        try:
            loss_dict = super().update()
        finally:
            self._amp_batches.close()
            self._amp_batches = None
            self._amp_normalization_batch = None
        # AMP_mjlab averages the discriminator losses over the mini-batches it actually optimized.
        updates = max(self._amp_updates, 1)
        mean_metrics = torch.stack(list(self._amp_totals.values())) / updates
        loss_dict.update(zip(self._amp_totals, mean_metrics.tolist(), strict=True))
        self.amp_dataset.clear_rollout()
        return loss_dict

    def _compute_auxiliary_loss(self) -> torch.Tensor:
        """Draw one policy/expert pair and compute the least-squares loss plus gradient penalty."""
        assert self._amp_batches is not None
        policy_batch, expert_batch = next(self._amp_batches)
        loss, metrics = self.discriminator.compute_loss(policy_batch, expert_batch, self.grad_pen_coef)
        # Retain raw features until the optimizer step completes; never feed normalized outputs
        # back into running statistics.
        self._amp_normalization_batch = (policy_batch, expert_batch)
        self._amp_updates += 1
        for name, value in metrics.items():
            self._amp_totals[name] = self._amp_totals.get(name, 0.0) + value
        return loss

    def _clip_gradients(self) -> None:
        """Clip the combined actor/critic norm, leaving discriminator gradients unclipped."""
        torch.nn.utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)

    def _after_optimizer_step(self) -> None:
        """Enforce the action-noise floor and merge raw frames into shared AMP statistics."""
        self._clamp_action_std()
        assert self._amp_normalization_batch is not None
        self.discriminator.update_normalization(*self._amp_normalization_batch)
        self._amp_normalization_batch = None

    def train_mode(self) -> None:
        """Enable training for PPO models and the discriminator normalizer."""
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        """Enable evaluation and freeze discriminator normalization statistics."""
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        """Include discriminator weights and normalization alongside the shared optimizer."""
        saved_dict = super().save()
        saved_dict["amp_discriminator_state_dict"] = self.discriminator.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load AMP states alongside the states selected by PPO.

        With no load_cfg, restore everything. The ``optimizer`` flag restores the shared AC/AMP
        optimizer, and ``discriminator`` controls discriminator weights and normalization.
        Actor-only checkpoints can initialize the policy without changing AMP.
        """
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        if load_cfg is None or load_cfg.get("discriminator"):
            self.discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> AMPPPO:
        """Resolve AMP metadata, then use PPO's model, storage, and algorithm construction.

        Configuration contains only constructor arguments. Large Dataset buffers are allocated
        in :meth:`__init__`, after RSL-RL resolves and copies the algorithm configuration.
        The task's reset loader remains independent of the training dataset.
        """
        algorithm_cfg = dict(cfg["algorithm"])
        amp_cfg = dict(algorithm_cfg.get("amp_cfg") or {})
        policy_key = amp_cfg.get("amp_policy_key", "amp_policy")
        base_env = cast(Any, env).unwrapped
        if not base_env.cfg.compute_final_obs:
            raise ValueError("AMP training requires compute_final_obs=True for terminal policy windows.")
        motion_cfg = base_env.cfg.events.init_amp_motion.params
        asset = base_env.scene[motion_cfg["asset_cfg"].name]
        amp_cfg["dataset_cfg"] = {
            "motion_dir": motion_cfg["motion_dir"],
            "recovery_dir": motion_cfg.get("recovery_dir"),
            "recovery_name_patterns": motion_cfg.get("recovery_name_patterns", ("fall", "getup", "get_up", "recovery")),
            "body_names": base_env.cfg.amp_body_names or asset.body_names,
            "joint_names": base_env.cfg.amp_joint_names or asset.joint_names,
            "anchor_name": base_env.cfg.amp_anchor_name,
            **amp_cfg.get("dataset_cfg", {}),
            "policy_body_names": asset.body_names,
            "policy_joint_names": asset.joint_names,
            "default_joint_pos": asset.data.default_joint_pos.torch[0],
            "default_joint_vel": asset.data.default_joint_vel.torch[0],
            "step_dt": base_env.step_dt,
        }
        algorithm_cfg["amp_cfg"] = amp_cfg
        # PPO stores only its observations, never the full raw AMP state.
        ppo_cfg = {**cfg, "algorithm": algorithm_cfg}
        ppo_cfg["obs_groups"] = {name: groups for name, groups in cfg["obs_groups"].items() if name != policy_key}
        alg = cast(AMPPPO, PPO.construct_algorithm(obs.exclude(policy_key), env, ppo_cfg, device))
        alg.amp_dataset.initialize_policy(obs[policy_key])
        return alg

    def broadcast_parameters(self) -> None:
        """Synchronize PPO and discriminator parameters and buffers from rank zero."""
        super().broadcast_parameters()
        for tensor in self.discriminator.state_dict().values():
            torch.distributed.broadcast(tensor, src=0)

    def reduce_parameters(self) -> None:
        """Average AC and discriminator gradients before the shared optimizer step."""
        super().reduce_parameters()
        reduce_gradients_in_buckets(self.discriminator.parameters(), self.gpu_world_size, self.grad_reduce_bucket_mb)

    def _configure_discriminator_optimizer(self, trunk_weight_decay: float, head_weight_decay: float) -> None:
        """Add the discriminator's trunk and head to PPO's shared optimizer."""
        head = self.discriminator.mlp[-1]
        head_parameter_ids = {id(parameter) for parameter in head.parameters()}
        trunk_parameters = [
            parameter for parameter in self.discriminator.parameters() if id(parameter) not in head_parameter_ids
        ]
        self.optimizer.add_param_group(
            {"params": trunk_parameters, "weight_decay": trunk_weight_decay, "name": "amp_trunk"}
        )
        self.optimizer.add_param_group(
            {"params": head.parameters(), "weight_decay": head_weight_decay, "name": "amp_head"}
        )

    def _resolve_min_std(self, value: float | Sequence[float] | None) -> torch.Tensor | None:
        """Resolve a scalar or per-action Gaussian noise floor."""
        if value is None:
            return None
        distribution = self._raw_actor.distribution
        if not isinstance(distribution, GaussianDistribution):
            raise ValueError("min_normalized_std requires a state-independent GaussianDistribution.")
        num_actions = distribution.output_dim
        floors = [float(value)] if isinstance(value, (int, float)) else [float(item) for item in value]
        if not floors or not all(math.isfinite(item) and item >= 0 for item in floors):
            raise ValueError(f"min_normalized_std must be nonnegative, got {value!r}.")
        if len(floors) == 1:
            floors *= num_actions
        if len(floors) != num_actions:
            raise ValueError(f"min_normalized_std requires one scalar or {num_actions} values, got {len(floors)}.")
        return torch.tensor(floors, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def _clamp_action_std(self) -> None:
        """Keep the actor's noise above ``min_normalized_std`` after an optimizer step."""
        if self.min_normalized_std is None:
            return
        distribution = self._raw_actor.distribution
        assert isinstance(distribution, GaussianDistribution)
        floor = self.min_normalized_std
        if distribution.std_type == "scalar":
            distribution.std_param.clamp_(min=floor)
        else:
            distribution.log_std_param.clamp_(min=torch.log(floor.clamp_min(1e-6)))
