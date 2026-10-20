"""AMP configuration for g1."""

import gymnasium as gym

from . import agents

gym.register(
    id="ZLab-AMP-Unitree-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:UnitreeG1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:G1AmpPPORunnerCfg",
    },
)

gym.register(
    id="ZLab-AMP-Unitree-G1-MoE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:UnitreeG1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:G1AmpMoEPPORunnerCfg",
    },
)

gym.register(
    id="ZLab-AMP-Unitree-G1-NoRecovery",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:UnitreeG1AmpNoRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:G1AmpNoRecoveryPPORunnerCfg",
    },
)
