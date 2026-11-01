"""Configuration checks for AMP-MJLab comparison tasks."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# Importing the task package performs Gym registration.
import z_lab.tasks  # noqa: F401
from z_lab.assets.robots.g1 import G1_CFG
from z_lab.assets.robots.g1_amp import G1_AMP_ACTION_SCALE, G1_AMP_CFG
from z_lab.tasks.amp.config.g1.agents.rsl_rl_amp_cfg import G1AmpPPORunnerCfg, G1AmpReferencePPORunnerCfg
from z_lab.tasks.amp.config.g1.amp_env_cfg import (
    UnitreeG1AmpEnvCfg,
    UnitreeG1AmpMjlabAssetEnvCfg,
    UnitreeG1AmpMjlabCommandsEnvCfg,
)
from z_lab.tasks.amp.mdp.curriculums import amp_velocity_ranges

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.mark.parametrize(
    ("task_id", "env_cfg_name", "agent_cfg_name"),
    [
        (
            "ZLab-AMP-Unitree-G1-MjlabCommands-Current",
            "UnitreeG1AmpMjlabCommandsEnvCfg",
            "G1AmpPPORunnerCfg",
        ),
        (
            "ZLab-AMP-Unitree-G1-MjlabCommands-Reference",
            "UnitreeG1AmpMjlabCommandsEnvCfg",
            "G1AmpReferencePPORunnerCfg",
        ),
        (
            "ZLab-AMP-Unitree-G1-MjlabAsset-Current",
            "UnitreeG1AmpMjlabAssetEnvCfg",
            "G1AmpPPORunnerCfg",
        ),
        (
            "ZLab-AMP-Unitree-G1-MjlabAsset-Reference",
            "UnitreeG1AmpMjlabAssetEnvCfg",
            "G1AmpReferencePPORunnerCfg",
        ),
    ],
)
def test_registered_amp_experiment_configs(task_id, env_cfg_name, agent_cfg_name):
    assert gym.spec(task_id).entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    assert type(env_cfg).__name__ == env_cfg_name
    assert type(agent_cfg).__name__ == agent_cfg_name


def test_amp_command_variants_and_curriculum_threshold():
    base_cfg = UnitreeG1AmpEnvCfg()
    assert base_cfg.curriculum is None
    assert not base_cfg.commands.base_velocity.heading_command

    for cfg_type in (UnitreeG1AmpMjlabCommandsEnvCfg, UnitreeG1AmpMjlabAssetEnvCfg):
        cfg = cfg_type()
        command = cfg.commands.base_velocity
        assert command.ranges.lin_vel_x == (-0.5, 1.0)
        assert command.ranges.lin_vel_y == (-0.5, 0.5)
        assert command.ranges.ang_vel_z == (-1.0, 1.0)
        assert command.heading_command
        assert command.rel_standing_envs == 0.05
        assert command.rel_heading_envs == 0.25
        assert command.resampling_time_range == (3.0, 8.0)

    command = object.__new__(UniformVelocityCommand)
    command.cfg = SimpleNamespace(
        ranges=SimpleNamespace(lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0))
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command), common_step_counter=120000)
    stages = UnitreeG1AmpMjlabCommandsEnvCfg().curriculum.velocity_ranges.params["velocity_stages"]
    amp_velocity_ranges(env, torch.empty(0, dtype=torch.long), "base_velocity", stages)
    assert command.cfg.ranges.lin_vel_x == (-0.5, 1.0)

    env.common_step_counter = 120001
    amp_velocity_ranges(env, torch.empty(0, dtype=torch.long), "base_velocity", stages)
    assert command.cfg.ranges.lin_vel_x == (-1.0, 2.0)
    assert command.cfg.ranges.lin_vel_y == (-1.0, 1.0)
    assert command.cfg.ranges.ang_vel_z == (-1.0, 1.0)


def test_amp_asset_isolated_and_complete():
    cfg = UnitreeG1AmpMjlabAssetEnvCfg()
    assert cfg.scene.robot is not G1_CFG
    assert cfg.scene.robot.soft_joint_pos_limit_factor == 0.9
    assert G1_CFG.soft_joint_pos_limit_factor == 1.0
    assert len(G1_AMP_ACTION_SCALE) == 29
    assert cfg.actions.joint_pos.scale == G1_AMP_ACTION_SCALE

    init = G1_AMP_CFG.init_state
    assert init.pos == (0.0, 0.0, 0.78)
    assert init.joint_pos[".*_hip_pitch_joint"] == -0.312
    assert init.joint_pos[".*_knee_joint"] == 0.669
    assert init.joint_pos[".*_ankle_pitch_joint"] == -0.363

    # Constructing AMP configs must not mutate the shared robot actuator parameters.
    shared_actuator = G1_CFG.actuators["N7520-14.3"]
    amp_actuator = G1_AMP_CFG.actuators["N7520-14.3"]
    assert shared_actuator.stiffness != amp_actuator.stiffness


@pytest.mark.parametrize(
    ("cfg_type", "discriminator_lr", "head_weight_decay", "replay_capacity"),
    [
        (G1AmpPPORunnerCfg, 1.0e-4, 1.0e-2, 500000),
        (G1AmpReferencePPORunnerCfg, 1.0e-3, 1.0e-1, 100000),
    ],
)
def test_amp_comparison_hyperparameters(cfg_type, discriminator_lr, head_weight_decay, replay_capacity):
    algorithm = cfg_type().algorithm
    assert algorithm.learning_rate == 1.0e-3
    assert algorithm.amp_cfg.discriminator_lr == discriminator_lr
    assert algorithm.amp_cfg.amp_head_weight_decay == head_weight_decay
    assert algorithm.amp_cfg.amp_replay_buffer_size == replay_capacity
