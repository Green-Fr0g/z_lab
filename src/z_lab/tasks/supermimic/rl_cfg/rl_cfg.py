"""Configuration templates for registered SuperMimic agents."""

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class SuperMimicWorldModelCfg:
    """Replay sequence sampling and supervised dynamics optimization."""

    learning_rate: float = 2e-4
    num_batches: int = 10
    batch_size: int = 2048
    horizon: int = 8
    nominal_root_position_error: float = 0.01
    nominal_root_orientation_error: float = 0.02
    nominal_joint_position_error: float = 0.1
    nominal_joint_velocity_error: float = 3.14
    nominal_body_position_error: float = 0.01
    nominal_body_orientation_error: float = 0.02
    nominal_body_linear_velocity_error: float = 0.1
    nominal_body_angular_velocity_error: float = 0.1
    max_grad_norm: float = 1.0


@configclass
class SuperMimicPolicyCfg:
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
class SuperMimicVelocityCfg:
    """Velocity encoder commands and training perturbations."""

    lin_vel_x_range: tuple[float, float] = (-1.5, 1.5)
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z_range: tuple[float, float] = (-3.14, 3.14)
    window_frames: int = 25
    command_noise: tuple[float, float, float] = (0.5, 0.5, 0.785)


@configclass
class SuperMimicProprioNoiseCfg:
    """Uniform noise half-widths for actor proprioception during training."""

    projected_gravity: float = 0.05
    base_ang_vel: float = 0.2
    joint_pos: float = 0.01
    joint_vel: float = 0.5
    actions: float = 0.0


@configclass
class SuperMimicReturnSamplerCfg:
    """Return-based reference state initialization settings."""

    gamma: float = 0.99
    horizon: int = 20
    memory: int = 256
    steepness: float = 25.0
    explore: float = 0.5
    keep: float = 0.1


@configclass
class SuperMimicRunnerCfg:
    """Common agent configuration populated by each registered task."""

    experiment_name: str = MISSING
    run_name: str = ""
    seed: int = 42
    representation: Literal["gencoord", "maxcoord"] = MISSING
    max_iterations: int = 1_000_000
    log_interval: int = 10
    save_interval: int = 1000
    replay_buffer_size: int = 1000
    replay_ratio: float = 16.0
    sampling_mode: Literal["return", "uniform", "start"] = "return"
    history_length: int = 1
    action_noise: float = 0.1
    dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    latent_dim: int = 32
    latent_bc_weight: float = 0.1
    slot_mix: dict[str, float] = {"motion": 0.5, "velocity": 0.5}
    world_model: SuperMimicWorldModelCfg = SuperMimicWorldModelCfg()
    policy: SuperMimicPolicyCfg = SuperMimicPolicyCfg()
    velocity: SuperMimicVelocityCfg = SuperMimicVelocityCfg()
    proprio_noise: SuperMimicProprioNoiseCfg = SuperMimicProprioNoiseCfg()
    return_sampler: SuperMimicReturnSamplerCfg = SuperMimicReturnSamplerCfg()
