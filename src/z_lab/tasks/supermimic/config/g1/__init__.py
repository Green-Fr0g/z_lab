"""G1 SuperMimic task registrations."""

import gymnasium as gym

gym.register(
    id="ZLab-SuperMimic-GenCoord-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1SuperMimicGenCoordEnvCfg",
        "supermimic_cfg_entry_point": f"{__name__}.agents.supermimic_cfg:G1SuperMimicGenCoordRunnerCfg",
    },
)

gym.register(
    id="ZLab-SuperMimic-MaxCoord-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1SuperMimicMaxCoordEnvCfg",
        "supermimic_cfg_entry_point": f"{__name__}.agents.supermimic_cfg:G1SuperMimicMaxCoordRunnerCfg",
    },
)
