"""Reusable RSL-RL configuration classes for AMP agents."""

from dataclasses import MISSING
from typing import Any

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class AmpCfg:
    """Configuration for the AMP discriminator and motion data."""

    amp_reward_coef: float = MISSING
    amp_replay_buffer_size: int = MISSING
    amp_discr_hidden_dims: list[int] = MISSING
    amp_discr_activation: str = MISSING
    discriminator_lr: float = MISSING
    discriminator_optimizer: str = MISSING
    amp_trunk_weight_decay: float = MISSING
    amp_head_weight_decay: float = MISSING
    grad_pen_coef: float = MISSING
    amp_task_reward_lerp: float = MISSING
    amp_state_key: str = MISSING
    motion_loader_class_name: str = MISSING
    motion_loader_kwargs: dict[str, Any] = MISSING


@configclass
class RslRlAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for an AMP algorithm based on PPO."""

    class_name: str = "AMPPPO"
    amp_cfg: AmpCfg = MISSING


@configclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for an on-policy runner using AMP."""

    algorithm: RslRlAmpAlgorithmCfg = MISSING
