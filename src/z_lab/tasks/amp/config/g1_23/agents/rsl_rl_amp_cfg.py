"""RSL-RL 5.5-compatible AMP-PPO settings for G1_23."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg

from z_lab.assets.robots.g1_23 import G1_23DOF_AMP_LINK_NAMES, G1_23DOF_ANCHOR_LINK_NAME
from z_lab.tasks.amp.config.g1_23.amp_env_cfg import MOTION_DIR, RECOVERY_CLIP_NAME
from z_lab.tasks.amp.rl_cfg import RslRlAMPPpoAlgorithmCfg


@configclass
class G1_23DOFAMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    """AMP-PPO runner config following amp_mjlab's policy and rollout sizes."""

    num_steps_per_env = 24
    max_iterations = 100_001
    save_interval = 100
    experiment_name = "g1_23_amp_locomotion"
    obs_groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlAMPPpoAlgorithmCfg(
        amp_motion_files=str(MOTION_DIR),
        amp_body_names=G1_23DOF_AMP_LINK_NAMES,
        amp_anchor_name=G1_23DOF_ANCHOR_LINK_NAME,
        amp_exclude_motion_names=(RECOVERY_CLIP_NAME,),
        amp_reward_coef=0.1,
        amp_task_reward_lerp=0.75,
        amp_replay_buffer_size=100_000,
        amp_discr_hidden_dims=[1024, 512, 256],
        amp_head_weight_decay=0.1,
        amp_min_std=0.05,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
