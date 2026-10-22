"""AMP agent configuration for the Unitree G1."""

from dataclasses import MISSING
from typing import Any

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from z_lab.assets.robots.g1 import G1_AMP_LINK_NAMES, G1_ANCHOR_LINK_NAME
from z_lab.tasks.amp.config.g1.amp_env_cfg import G1_AMP_ALL_BODY_NAMES, G1_AMP_MOTION_DIR


@configclass
class AmpCfg:
    """Configuration for the AMP discriminator and motion data."""

    amp_reward_coef: float = MISSING
    """The coefficient used to scale the AMP reward."""

    amp_replay_buffer_size: int = MISSING
    """The capacity of the AMP replay buffer."""

    amp_discr_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the AMP discriminator."""

    amp_discr_activation: str = MISSING
    """The activation function used by the AMP discriminator."""

    discriminator_lr: float = MISSING
    """The learning rate of the AMP discriminator."""

    discriminator_optimizer: str = MISSING
    """The optimizer used for the AMP discriminator."""

    amp_trunk_weight_decay: float = MISSING
    """The weight decay applied to the discriminator trunk."""

    amp_head_weight_decay: float = MISSING
    """The weight decay applied to the discriminator head."""

    grad_pen_coef: float = MISSING
    """The coefficient of the discriminator gradient penalty."""

    amp_task_reward_lerp: float = MISSING
    """The interpolation factor between task and AMP rewards."""

    amp_state_key: str = MISSING
    """The observation key used for the AMP expert state."""

    motion_loader_class_name: str = MISSING
    """The import path of the motion loader class."""

    motion_loader_kwargs: dict[str, Any] = MISSING
    """The keyword arguments passed to the motion loader."""


@configclass
class RslRlAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for an AMP algorithm based on PPO."""

    class_name: str = "AMPPPO"
    """The AMP algorithm class name."""

    amp_cfg: AmpCfg = MISSING
    """The AMP-specific algorithm configuration."""


@configclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for an on-policy runner using AMP."""

    algorithm: RslRlAmpAlgorithmCfg = MISSING
    """The AMP algorithm configuration."""


@configclass
class G1AmpPPORunnerCfg(RslRlAmpRunnerCfg):
    """AMP-PPO runner for the Unitree G1 flat locomotion task."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "g1_amp_flat"

    obs_groups = {"actor": ["policy"], "critic": ["critic"], "amp_state": ["amp_state"]}
    """Policy, critic, and discriminator inputs available from the environment."""

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlAmpAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=AmpCfg(
            amp_reward_coef=0.1,
            amp_replay_buffer_size=2000000,
            amp_discr_hidden_dims=[1024, 512],
            amp_discr_activation="relu",
            discriminator_lr=1.0e-3,
            discriminator_optimizer="adam",
            amp_trunk_weight_decay=1.0e-3,
            amp_head_weight_decay=1.0e-2,
            grad_pen_coef=10.0,
            amp_task_reward_lerp=0.75,
            amp_state_key="amp_state",
            motion_loader_class_name="z_lab.tasks.amp.utils.motion_loader:AMPBodyStateMotionLoader",
            motion_loader_kwargs={
                "motion_files": G1_AMP_MOTION_DIR,
                "body_names": tuple(G1_AMP_LINK_NAMES),
                "anchor_name": G1_ANCHOR_LINK_NAME,
                "all_body_names": G1_AMP_ALL_BODY_NAMES,
                "quat_order": "xyzw",
            },
        ),
    )
