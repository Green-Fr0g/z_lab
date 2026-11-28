"""Configuration templates for registered SuperAmp agents."""

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class SuperAmpWorldModelCfg:
    """Replay sequence sampling and supervised dynamics optimization."""

    learning_rate: float = 2e-4
    num_batches: int = 10
    batch_size: int = 2048
    horizon: int = 8
    nominal_root_position_error: float = 0.01
    nominal_root_orientation_error: float = 0.02
    nominal_joint_position_error: float = 0.1
    nominal_joint_velocity_error: float = 3.14
    nominal_body_position_error: float = 0.05
    nominal_body_orientation_error: float = 0.1
    nominal_body_linear_velocity_error: float = 1.0
    nominal_body_angular_velocity_error: float = 3.14
    max_grad_norm: float = 1.0
    compile: bool = False


@configclass
class SuperAmpPolicyCfg:
    """Policy rollout and learning settings."""

    kind: Literal["actor-critic", "bptt"] = "actor-critic"
    learning_rate: float = 2e-4
    num_batches: int = 1
    batch_size: int = 4096
    horizon: int = 32
    gamma: float | None = None
    """Use the selected learner's discount when unset (0.99 for actor-critic, 1.0 for BPTT)."""
    max_grad_norm: float = 1.0
    survival_masking: bool = True
    beta2: float = 0.95
    grad_clip_value: float = 0.1
    lam: float = 0.95
    critic_updates: int = 5
    critic_mini_batches: int = 4


@configclass
class SuperAmpDiscriminatorCfg:
    """Discriminator optimization and AMP sampling window."""

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


@configclass
class SuperAmpProprioNoiseCfg:
    """Independent uniform proprioception noise half-widths."""

    projected_gravity: float = 0.05
    base_ang_vel: float = 0.2
    joint_pos: float = 0.01
    joint_vel: float = 0.5
    actions: float = 0.0


@configclass
class SuperAmpRunnerCfg:
    """Agent configuration for registered SuperAMP velocity tasks."""

    experiment_name: str = MISSING
    run_name: str = ""
    seed: int = 42
    representation: Literal["gencoord", "maxcoord"] = MISSING
    max_iterations: int = 1_000_000
    log_interval: int = 10
    save_interval: int = 1000
    replay_buffer_size: int = 1000
    amp_replay_buffer_size: int = 200_000
    warmup_steps: int = 16
    replay_ratio: float = 16.0
    history_length: int = 1
    action_noise: float = 0.1
    dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    world_model: SuperAmpWorldModelCfg = SuperAmpWorldModelCfg()
    policy: SuperAmpPolicyCfg = SuperAmpPolicyCfg()
    proprio_noise: SuperAmpProprioNoiseCfg = SuperAmpProprioNoiseCfg()
    discriminator: SuperAmpDiscriminatorCfg = SuperAmpDiscriminatorCfg()
