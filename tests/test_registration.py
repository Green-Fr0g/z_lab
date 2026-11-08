"""Tests for the project's task registrations."""

import gymnasium as gym

import z_lab.tasks  # noqa: F401


def test_task_registrations():
    expected = {
        "ZLab-Velocity-Unitree-G1": "UnitreeG1VelocityEnvCfg",
        "ZLab-Velocity-Unitree-G1-23DOF": "UnitreeG123VelocityEnvCfg",
        "ZLab-Velocity-DR02": "DR02VelocityEnvCfg",
        "ZLab-AMP-Unitree-G1-23DOF": "UnitreeG123AmpEnvCfg",
    }
    for task_id, class_name in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
        assert spec.kwargs["env_cfg_entry_point"].endswith(f":{class_name}")

    amp_spec = gym.spec("ZLab-AMP-Unitree-G1-23DOF")
    assert amp_spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":G1_23DOFAMPRunnerCfg")
