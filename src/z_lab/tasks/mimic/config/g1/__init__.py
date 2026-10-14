"""G1 Mimic task registrations."""

import gymnasium as gym

from . import agents

gym.register(
    id="ZLab-Mimic-Unitree-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1MimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MimicPPORunnerCfg",
    },
)
gym.register(
    id="ZLab-Mimic-Unitree-G1-Wo-State-Estimation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1MimicWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:G1MimicWoStateEstimationPPORunnerCfg"),
    },
)
