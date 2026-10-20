# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity commands for AMP locomotion training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class AmpVelocityCommand(UniformVelocityCommand):
    """SE(2) velocity command generator with a turn-only environment fraction.

    Builds directly on :class:`~isaaclab.envs.mdp.UniformVelocityCommand`, so the observation and
    reward contract stays the standard ``[lin_vel_x, lin_vel_y, ang_vel_z]`` in the robot base frame.

    The only behavioral addition is a fraction of environments that receive a yaw command only.
    Those environments get their linear command zeroed and keep the yaw command already sampled from
    the regular :attr:`~isaaclab.envs.mdp.UniformVelocityCommandCfg.ranges` ``ang_vel_z`` range, so
    no separate yaw range is introduced.

    Every environment is assigned exactly one role per resample: turn-only, standing, or the regular
    full command. The two special masks are sampled as disjoint partitions of the same draw, so an
    environment can never be turn-only and standing at the same time and their sizes match the
    configured probabilities exactly.
    """

    cfg: AmpVelocityCommandCfg

    def __init__(self, cfg: AmpVelocityCommandCfg, env):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # per-env role flag, resampled together with the command
        self.is_turn_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"\n\tTurn-only probability: {self.cfg.rel_turn_envs}"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        # resample the standard command, standing and heading flags included
        super()._resample_command(env_ids)
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        if num_envs == 0:
            return
        # Partition the sample into disjoint roles. A single draw is split into
        # [turn-only | standing | regular], so the masks can never overlap and both
        # probabilities are honored exactly.
        role = torch.rand(num_envs, device=self.device)
        standing_threshold = self.cfg.rel_turn_envs + self.cfg.rel_standing_envs
        self.is_turn_env[env_ids] = role < self.cfg.rel_turn_envs
        self.is_standing_env[env_ids] = (role >= self.cfg.rel_turn_envs) & (role < standing_threshold)

    def _update_command(self):
        """Zero the linear command for turn-only environments, then apply the base update."""
        self.vel_command_b[self.is_turn_env, :2] = 0.0
        super()._update_command()


@configclass
class AmpVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the AMP velocity command generator."""

    class_type: type = AmpVelocityCommand

    rel_turn_envs: float = 0.2
    """The sampled probability of environments that only receive a yaw command.

    For these environments ``lin_vel_x`` and ``lin_vel_y`` are forced to exactly zero, while the yaw
    command keeps the value sampled from :attr:`~isaaclab.envs.mdp.UniformVelocityCommandCfg.Ranges`
    ``ang_vel_z``. Turn-only and standing environments are disjoint. Defaults to 0.2.
    """

    def __post_init__(self):
        if not 0.0 <= self.rel_turn_envs <= 1.0:
            raise ValueError(f"The turn-only probability must be in [0, 1], got {self.rel_turn_envs}.")
        if self.rel_turn_envs + self.rel_standing_envs > 1.0:
            raise ValueError(
                "The turn-only and standing probabilities must not overlap, got"
                f" rel_turn_envs={self.rel_turn_envs} and rel_standing_envs={self.rel_standing_envs}."
            )

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        """Uniform distribution ranges plus the AMP command curriculum limits."""

        limit_lin_vel_x: tuple[float, float] = MISSING
        """Lower and upper limits the ``lin_vel_x`` range may reach through the command curriculum."""

        limit_lin_vel_y: tuple[float, float] = MISSING
        """Lower and upper limits the ``lin_vel_y`` range may reach through the command curriculum."""

        limit_ang_vel_z: tuple[float, float] = MISSING
        """Lower and upper limits the ``ang_vel_z`` range may reach through the command curriculum."""

    ranges: Ranges = MISSING
    """Command sampling ranges including the AMP curriculum limits."""
