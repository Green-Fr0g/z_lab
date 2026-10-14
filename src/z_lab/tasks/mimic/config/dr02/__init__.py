"""DR02 Mimic task registrations."""

import gymnasium as gym

from . import agents

gym.register(
    id="ZLab-Mimic-DR02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:DR02MimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DR02MimicPPORunnerCfg",
    },
)
gym.register(
    id="ZLab-Mimic-DR02-Wo-State-Estimation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:DR02MimicWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:DR02MimicWoStateEstimationPPORunnerCfg"),
    },
)
