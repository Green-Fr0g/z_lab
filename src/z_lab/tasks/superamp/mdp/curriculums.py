"""Linear and angular velocity command curricula for SuperAMP locomotion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def velocity_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice,
    lin_vel_reward_term_name: str = "track_lin_vel_xy",
    ang_vel_reward_term_name: str = "track_ang_vel_z",
) -> dict[str, torch.Tensor]:
    """Expand planar velocity and yaw-rate commands using their respective tracking rewards.

    At global episode-length boundaries, a mean episode reward per second above
    80% of the corresponding reward weight expands both ends of its command
    range by 0.1 m/s for planar velocity or 0.1 rad/s for yaw rate.
    Return the upper forward velocity and yaw-rate bounds for logging.
    """
    command_cfg = env.command_manager.get_term("base_velocity").cfg
    ranges = command_cfg.ranges
    limit_ranges = command_cfg.limit_ranges
    reward_manager = env.reward_manager

    if env.common_step_counter % env.max_episode_length == 0:
        lin_vel_weight = reward_manager.get_term_cfg(lin_vel_reward_term_name).weight
        lin_vel_reward = (
            reward_manager._episode_sums[lin_vel_reward_term_name][env_ids].mean() / env.max_episode_length_s
        )
        if lin_vel_reward > 0.8 * lin_vel_weight:
            x_min, x_max = limit_ranges.lin_vel_x
            y_min, y_max = limit_ranges.lin_vel_y
            ranges.lin_vel_x = (
                min(max(ranges.lin_vel_x[0] - 0.1, x_min), x_max),
                min(max(ranges.lin_vel_x[1] + 0.1, x_min), x_max),
            )
            ranges.lin_vel_y = (
                min(max(ranges.lin_vel_y[0] - 0.1, y_min), y_max),
                min(max(ranges.lin_vel_y[1] + 0.1, y_min), y_max),
            )

        ang_vel_weight = reward_manager.get_term_cfg(ang_vel_reward_term_name).weight
        ang_vel_reward = (
            reward_manager._episode_sums[ang_vel_reward_term_name][env_ids].mean() / env.max_episode_length_s
        )
        if ang_vel_reward > 0.8 * ang_vel_weight:
            yaw_min, yaw_max = limit_ranges.ang_vel_z
            ranges.ang_vel_z = (
                min(max(ranges.ang_vel_z[0] - 0.1, yaw_min), yaw_max),
                min(max(ranges.ang_vel_z[1] + 0.1, yaw_min), yaw_max),
            )

    return {
        "lin_vel_x_max": torch.tensor(ranges.lin_vel_x[1], device=env.device),
        "ang_vel_z_max": torch.tensor(ranges.ang_vel_z[1], device=env.device),
    }
