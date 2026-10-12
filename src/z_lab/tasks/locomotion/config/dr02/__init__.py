import gymnasium as gym

from . import agents

gym.register(
    id="ZLab-Velocity-DR02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DR02VelocityEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DR02LocomtionPPORunnerCfg",
    },
)
