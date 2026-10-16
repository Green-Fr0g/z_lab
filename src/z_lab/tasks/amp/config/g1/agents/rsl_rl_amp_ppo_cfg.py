"""AMP-PPO agent configurations for MoE and ordinary MLP policies."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlMoEModelCfg(RslRlMLPModelCfg):
    """MLP-compatible model configuration with a learned expert gate."""

    class_name: str = "MoEModel"
    num_experts: int = 3
    gate_hidden_dims: list[int] = []


@configclass
class RslRlAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO options extended with online discriminator training settings."""

    class_name: str = "AMPPPO"
    amp_cfg: dict = {
        "amp_policy_key": "amp_policy",
        "amp_expert_key": "amp_expert",
        "discriminator_cfg": {
            "hidden_dims": [1024, 512, 256],
            "activation": "relu",
            "obs_normalization": True,
            "amp_reward_coef": 0.1,
            "task_reward_lerp": 0.75,
        },
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "trunk_weight_decay": 1e-3,
        "head_weight_decay": 1e-2,
        "grad_pen_coef": 10.0,
        # Epoch and mini-batch counts inherit the PPO values when omitted.
    }


@configclass
class G1AmpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Use the standard runner with independently configurable actor and critic."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "g1_amp"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMoEModelCfg(
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
    algorithm = RslRlAmpPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1AmpMLPPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Use ordinary MLP actor and critic models with the same online AMP-PPO setup."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "g1_amp_mlp"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
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
    algorithm = RslRlAmpPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
