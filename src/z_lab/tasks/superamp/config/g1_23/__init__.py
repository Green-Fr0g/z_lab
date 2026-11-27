"""G1 23-DOF SuperAMP task registrations."""

import gymnasium as gym

gym.register(
    id="ZLab-SuperAMP-GenCoord-G1-23DOF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.superamp_env_cfg:G1_23DOFSuperAmpGenCoordEnvCfg",
        "superamp_cfg_entry_point": f"{__name__}.agents.superamp_cfg:G1_23DOFSuperAmpGenCoordRunnerCfg",
    },
)

gym.register(
    id="ZLab-SuperAMP-MaxCoord-G1-23DOF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.superamp_env_cfg:G1_23DOFSuperAmpMaxCoordEnvCfg",
        "superamp_cfg_entry_point": f"{__name__}.agents.superamp_cfg:G1_23DOFSuperAmpMaxCoordRunnerCfg",
    },
)
