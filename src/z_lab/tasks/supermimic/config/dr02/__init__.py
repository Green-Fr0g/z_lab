"""DR02 SuperMimic task registrations."""

import gymnasium as gym

gym.register(
    id="ZLab-SuperMimic-GenCoord-DR02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:DR02SuperMimicGenCoordEnvCfg",
        "supermimic_cfg_entry_point": f"{__name__}.agents.supermimic_cfg:DR02SuperMimicGenCoordRunnerCfg",
    },
)

gym.register(
    id="ZLab-SuperMimic-MaxCoord-DR02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:DR02SuperMimicMaxCoordEnvCfg",
        "supermimic_cfg_entry_point": f"{__name__}.agents.supermimic_cfg:DR02SuperMimicMaxCoordRunnerCfg",
    },
)
