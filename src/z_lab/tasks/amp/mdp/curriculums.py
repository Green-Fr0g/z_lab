"""Curricula for AMP locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def amp_velocity_ranges(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict[str, object]],
) -> dict[str, float]:
    """Expand velocity command ranges at configured global environment steps."""
    del env_ids
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, UniformVelocityCommand):
        raise TypeError(f"Command '{command_name}' must be a UniformVelocityCommand")

    for stage in velocity_stages:
        if env.common_step_counter > int(stage["step"]):
            for range_name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                value = stage.get(range_name)
                if value is not None:
                    setattr(command.cfg.ranges, range_name, value)

    ranges = command.cfg.ranges
    return {
        "lin_vel_x_min": float(ranges.lin_vel_x[0]),
        "lin_vel_x_max": float(ranges.lin_vel_x[1]),
        "lin_vel_y_min": float(ranges.lin_vel_y[0]),
        "lin_vel_y_max": float(ranges.lin_vel_y[1]),
        "ang_vel_z_min": float(ranges.ang_vel_z[0]),
        "ang_vel_z_max": float(ranges.ang_vel_z[1]),
    }
