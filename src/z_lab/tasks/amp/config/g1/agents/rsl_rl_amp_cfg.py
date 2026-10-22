from dataclasses import field

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlAmpDatasetCfg:
    """Shared feature and sampling configuration; robot metadata is resolved from the environment."""

    num_frames: int = 2
    """Number of frames per discriminator input, ordered from oldest to newest."""

    history_stride: int = 1
    """Control steps between consecutive selected frames."""

    recovery_sample_ratio: float = 0.2
    """Recovery fraction of each expert batch."""

    replay_buffer_size: int = 0
    """Policy replay capacity; zero uses only current-rollout samples."""

    obs_string: list[str] = field(
        default_factory=lambda: ["body_pos_b", "body_ori_b", "body_lin_vel_b", "body_ang_vel_b"]
    )
    """Ordered registered features, shared by expert and policy. Anchor gravity uses body_gravity_b."""

    joint_pos_scale: float = 1.0
    """Multiplier applied to joint positions relative to the robot's defaults."""

    joint_vel_scale: float = 0.05
    """Multiplier applied to joint velocities relative to the robot's defaults."""


@configclass
class RslRlAmpDiscriminatorCfg:
    """Configuration for the discriminator model; feature dimensions are inferred from the Dataset."""

    class_name: str = "rsl_rl.models:AMPDiscriminator"
    hidden_dims: list[int] = field(default_factory=lambda: [1024, 512, 256])
    activation: str = "relu"
    obs_normalization: bool = True
    obs_eps: float = 1e-2
    obs_clip: float | None = 10.0
    amp_reward_coef: float = 1.0
    """Scale of the quadratic imitation reward before mixing with task rewards."""

    task_reward_lerp: float = 0.0
    """Task reward fraction in [0, 1]."""


@configclass
class RslRlAmpCfg:
    """AMP extension configuration, serialized to dictionaries for RSL-RL."""

    amp_policy_key: str = "amp_policy"
    """Environment observation group containing packed, single-frame body/joint states."""

    dataset_cfg: RslRlAmpDatasetCfg = field(default_factory=RslRlAmpDatasetCfg)
    discriminator_cfg: RslRlAmpDiscriminatorCfg = field(default_factory=RslRlAmpDiscriminatorCfg)
    grad_pen_coef: float = 10.0
    trunk_weight_decay: float = 1e-3
    head_weight_decay: float = 1e-1
    min_normalized_std: float | list[float] | None = None
    """Scalar or per-action floor for a state-independent Gaussian policy's standard deviation."""


@configclass
class RslRlAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO that also trains an AMP discriminator on the motion clips."""

    class_name: str = "AMPPPO"

    amp_cfg: RslRlAmpCfg = field(default_factory=RslRlAmpCfg)
    """AMP settings passed to :class:`AMPPPO`.

    ``dataset_cfg`` selects shared features and temporal sampling. The algorithm constructs the
    dataset from motion paths and robot metadata; the environment only produces raw policy frames.
    Replay batches follow PPO's mini-batch size and update count. The discriminator shares PPO's
    optimizer and learning-rate schedule. ``obs_string`` controls the ordered features of both
    expert and policy inputs; normalization statistics are shared across temporal positions.
    """


@configclass
class RslRlMoEModelCfg(RslRlMLPModelCfg):
    """Dense mixture of experts with the same observation/distribution interface as an MLP."""

    class_name: str = "MoEModel"
    num_experts: int = 3
    gate_hidden_dims: list[int] = field(default_factory=list)


@configclass
class G1AmpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_amp"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm: RslRlAmpPpoAlgorithmCfg = RslRlAmpPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            amp_policy_key="amp_policy",
            dataset_cfg=RslRlAmpDatasetCfg(
                num_frames=2,
                history_stride=1,
                recovery_sample_ratio=0.2,
                replay_buffer_size=100_000,
                # Shared expert/policy feature order; optional anchor gravity: "body_gravity_b".
                obs_string=["body_pos_b", "body_ori_b", "body_lin_vel_b", "body_ang_vel_b"],
            ),
            discriminator_cfg=RslRlAmpDiscriminatorCfg(
                hidden_dims=[1024, 512, 256],
                activation="relu",
                obs_normalization=True,
                task_reward_lerp=0.75,
                amp_reward_coef=0.1,
            ),
            grad_pen_coef=10.0,
            trunk_weight_decay=1e-3,
            head_weight_decay=1e-1,
            min_normalized_std=0.05,
        ),
    )


@configclass
class G1AmpMoEPPORunnerCfg(G1AmpPPORunnerCfg):
    """AMP with recovery and MoE actor/critic; the discriminator remains an MLP."""

    experiment_name = "g1_amp_moe"
    actor = RslRlMoEModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        num_experts=3,
        gate_hidden_dims=[],
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMoEModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        num_experts=3,
        gate_hidden_dims=[],
    )


@configclass
class G1AmpNoRecoveryPPORunnerCfg(G1AmpPPORunnerCfg):
    """MLP AMP baseline using only locomotion expert samples."""

    experiment_name = "g1_amp_no_recovery"

    def __post_init__(self) -> None:
        self.algorithm.amp_cfg.dataset_cfg.recovery_sample_ratio = 0.0
