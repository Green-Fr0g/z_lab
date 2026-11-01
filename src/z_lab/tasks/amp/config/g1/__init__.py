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

for task_suffix, env_cfg in (
    ("MjlabCommands", "UnitreeG1AmpMjlabCommandsEnvCfg"),
    ("MjlabAsset", "UnitreeG1AmpMjlabAssetEnvCfg"),
):
    for variant, agent_cfg in (
        ("Current", "G1AmpPPORunnerCfg"),
        ("Reference", "G1AmpReferencePPORunnerCfg"),
    ):
        gym.register(
            id=f"ZLab-AMP-Unitree-G1-{task_suffix}-{variant}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.amp_env_cfg:{env_cfg}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:{agent_cfg}",
            },
        )
