"""G1 23-DOF SuperMimic task registrations."""

import gymnasium as gym

gym.register(
    id="ZLab-SuperMimic-GenCoord-G1-23DOF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1_23DOFSuperMimicGenCoordEnvCfg",
        "supermimic_cfg_entry_point": f"{__name__}.agents.supermimic_cfg:G1_23DOFSuperMimicGenCoordRunnerCfg",
    },
)

gym.register(
    id="ZLab-SuperMimic-MaxCoord-G1-23DOF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1_23DOFSuperMimicMaxCoordEnvCfg",
        "supermimic_cfg_entry_point": f"{__name__}.agents.supermimic_cfg:G1_23DOFSuperMimicMaxCoordRunnerCfg",
    },
)
