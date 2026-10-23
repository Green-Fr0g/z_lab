"""AMP agent configuration for the Unitree G1."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from z_lab.assets.robots.g1 import G1_AMP_LINK_NAMES, G1_ANCHOR_LINK_NAME
from z_lab.tasks.amp.config.g1.amp_env_cfg import G1_AMP_ALL_BODY_NAMES, G1_AMP_MOTION_DIR
from z_lab.tasks.amp.rl_cfg import AmpCfg, RslRlAmpAlgorithmCfg, RslRlAmpRunnerCfg


@configclass
class G1AmpPPORunnerCfg(RslRlAmpRunnerCfg):
    """AMP-PPO runner for the Unitree G1 flat locomotion task."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "g1_amp"

    obs_groups = {"actor": ["policy"], "critic": ["critic"], "amp_state": ["amp_state"]}
    """Policy, critic, and discriminator inputs available from the environment."""

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
    algorithm = RslRlAmpAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
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
            amp_replay_buffer_size=500000,
            amp_discr_hidden_dims=[1024, 512, 256],
            amp_discr_activation="relu",
            discriminator_lr=1.0e-4,
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
