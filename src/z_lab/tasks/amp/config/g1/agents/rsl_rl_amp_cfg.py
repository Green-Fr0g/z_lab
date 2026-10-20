from dataclasses import field

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO that also trains an AMP discriminator on the motion clips."""

    class_name: str = "AMPPPO"

    amp_cfg: dict = field(default_factory=dict)
    """AMP settings passed to :class:`AMPPPO`.

    ``dataset_cfg`` selects shared features and temporal sampling. The algorithm constructs the
    dataset from motion paths and robot metadata; the environment only produces raw policy frames.
    AC and discriminator updates share the PPO epochs, mini-batches, optimizer and learning rate.
    """


@configclass
class RslRlMoEModelCfg(RslRlMLPModelCfg):
    """Dense mixture of experts with the same observation/distribution interface as an MLP."""

    class_name: str = "rsl_rl.models:MoEModel"
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
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
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
        amp_cfg={
            "amp_policy_key": "amp_policy",
            "dataset_cfg": {
                "num_frames": 4,
                "history_stride": 2,
                "recovery_sample_ratio": 0.2,
                "obs_string": ["body_pos_b", "body_ori_b", "body_lin_vel_b", "body_ang_vel_b", "body_gravity_b"],
            },
            "discriminator_cfg": {
                "hidden_dims": [1024, 512, 256],
                "activation": "elu",
                "obs_normalization": True,
                # blend the task reward with the style reward so tracking still drives the gait
                "task_reward_lerp": 0.75,
                "amp_reward_coef": 2.0,
            },
            "grad_pen_coef": 10.0,
            "trunk_weight_decay": 1e-3,
            "head_weight_decay": 1e-2,
        },
    )


@configclass
class G1AmpMoEPPORunnerCfg(G1AmpPPORunnerCfg):
    """AMP with recovery and MoE actor/critic; the discriminator remains an MLP."""

    experiment_name = "g1_amp_moe"
    actor = RslRlMoEModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        num_experts=3,
        gate_hidden_dims=[],
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMoEModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        num_experts=3,
        gate_hidden_dims=[],
    )


@configclass
class G1AmpNoRecoveryPPORunnerCfg(G1AmpPPORunnerCfg):
    """MLP AMP baseline using only locomotion expert samples."""

    experiment_name = "g1_amp_no_recovery"

    def __post_init__(self) -> None:
        self.algorithm.amp_cfg["dataset_cfg"]["recovery_sample_ratio"] = 0.0
