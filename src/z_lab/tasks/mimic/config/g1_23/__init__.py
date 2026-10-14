"""G1 23-DOF Mimic task registrations."""

import gymnasium as gym

from . import agents

gym.register(
    id="ZLab-Mimic-Unitree-G1-23DOF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1_23DOFMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_23DOFMimicPPORunnerCfg",
    },
)
gym.register(
    id="ZLab-Mimic-Unitree-G1-23DOF-Wo-State-Estimation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1_23DOFMimicWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:G1_23DOFMimicWoStateEstimationPPORunnerCfg"),
    },
)
