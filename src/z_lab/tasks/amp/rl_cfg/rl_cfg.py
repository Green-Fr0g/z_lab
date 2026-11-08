"""Reusable RSL-RL algorithm configuration for AMP tasks."""

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


@configclass
class RslRlAMPPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Describe the AMP-PPO constructor arguments consumed by the RSL-RL runner."""

    class_name: str = "rsl_rl.algorithms:AMPPPO"
    """Algorithm entry point resolved and instantiated by the runner."""

    amp_motion_files: str = MISSING
    """Path to the expert motion NPZ files or their directory."""

    amp_body_names: tuple[str, ...] = MISSING
    """Ordered body names used to construct the AMP features."""

    amp_anchor_name: str = MISSING
    """Body defining the reference frame for the AMP features."""

    amp_exclude_motion_names: tuple[str, ...] = ()
    """Motion clip names excluded from discriminator expert sampling."""

    amp_reward_coef: float = 0.1
    """Scale of the discriminator's motion reward."""

    amp_task_reward_lerp: float = 0.75
    """Task reward fraction in the combined task and motion reward."""

    amp_replay_buffer_size: int = 100_000
    """Maximum number of motion transitions stored in the replay buffer."""

    amp_discr_hidden_dims: list[int] = [1024, 512, 256]
    """Hidden layer sizes of the motion discriminator."""

    amp_head_weight_decay: float = 0.1
    """Weight decay applied to the discriminator's output layer."""

    amp_min_std: float = 0.05
    """Lower bound of the actor's action standard deviation."""
