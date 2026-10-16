"""Unitree G1 AMP task registrations."""

import gymnasium as gym

gym.register(
    id="ZLab-AMP-Unitree-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:G1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_amp_ppo_cfg:G1AmpPPORunnerCfg",
        "rsl_rl_amp_mlp_cfg_entry_point": f"{__name__}.agents.rsl_rl_amp_ppo_cfg:G1AmpMLPPPORunnerCfg",
    },
)

gym.register(
    id="ZLab-AMP-Unitree-G1-No-Recovery",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:G1AmpNoRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_amp_ppo_cfg:G1AmpPPORunnerCfg",
        "rsl_rl_amp_mlp_cfg_entry_point": f"{__name__}.agents.rsl_rl_amp_ppo_cfg:G1AmpMLPPPORunnerCfg",
    },
)

gym.register(
    id="ZLab-AMP-Unitree-G1-MoE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:G1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_amp_ppo_cfg:G1AmpPPORunnerCfg",
    },
)
