"""Register the AMP locomotion task for G1 29-DOF."""

import gymnasium as gym

from . import agents, amp_env_cfg

gym.register(
    id="ZLab-AMP-Unitree-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{amp_env_cfg.__name__}:UnitreeG1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:G1AMPRunnerCfg",
    },
)
