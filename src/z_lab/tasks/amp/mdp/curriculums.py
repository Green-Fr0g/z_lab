# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP-specific curricula."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_anchor_linear_velocity",
) -> torch.Tensor:
    """Widen the linear command range when linear velocity tracking saturates."""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    # avoid updating the command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # if the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # clamp to ensure we don't exceed the final ranges
            new_vel_x = torch.clamp(
                new_vel_x, min=base_velocity_ranges.limit_lin_vel_x[0], max=base_velocity_ranges.limit_lin_vel_x[1]
            )
            new_vel_y = torch.clamp(
                new_vel_y, min=base_velocity_ranges.limit_lin_vel_y[0], max=base_velocity_ranges.limit_lin_vel_y[1]
            )

            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_anchor_angular_velocity",
) -> torch.Tensor:
    """Widen the yaw command range when yaw tracking saturates."""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    # avoid updating the command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # if the tracking reward is above 60% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.6 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # clamp to ensure we don't exceed the final range
            new_ang_vel_z = torch.clamp(
                new_ang_vel_z,
                min=base_velocity_ranges.limit_ang_vel_z[0],
                max=base_velocity_ranges.limit_ang_vel_z[1],
            )

            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)
