"""Velocity commands with dedicated lateral and in-place turning samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

from z_lab.tasks.amp.mdp.commands import UniformLevelVelocityCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SuperAmpVelocityCommand(UniformVelocityCommand):
    """Sample mixed, standing, lateral, and in-place turning velocity commands."""

    cfg: SuperAmpVelocityCommandCfg

    def __init__(self, cfg: SuperAmpVelocityCommandCfg, env: ManagerBasedEnv):
        probabilities = (cfg.rel_standing_envs, cfg.rel_lateral_envs, cfg.rel_turning_envs)
        if any(not 0.0 <= probability <= 1.0 for probability in probabilities) or sum(probabilities) > 1.0:
            raise ValueError("Standing, lateral, and turning probabilities must be in [0, 1] and sum to at most 1.")
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        super()._resample_command(env_ids)
        lateral_probability = self.cfg.rel_lateral_envs
        turning_probability = self.cfg.rel_turning_envs
        if lateral_probability == 0.0 and turning_probability == 0.0:
            return

        sampled_velocity = self.vel_command_b[env_ids]
        moving = ~self.is_standing_env[env_ids]
        # The base sampler already assigned standing commands. Scale the draw by the remaining probability mass.
        draw = torch.rand_like(sampled_velocity[:, 0]) * (1.0 - self.cfg.rel_standing_envs)
        lateral = moving & (draw < lateral_probability)
        turning = moving & (draw >= lateral_probability) & (draw < lateral_probability + turning_probability)

        zero_components = torch.stack((lateral | turning, turning, lateral), dim=-1)
        self.vel_command_b[env_ids] = sampled_velocity.masked_fill(zero_components, 0.0)
        self.is_heading_env[env_ids] &= ~(lateral | turning)


@configclass
class SuperAmpVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Configure the velocity ranges, curriculum limits, and command sampling probabilities."""

    class_type: type = SuperAmpVelocityCommand

    rel_lateral_envs: float = 0.0
    """Probability of sampling a pure lateral command from the current y velocity range."""

    rel_turning_envs: float = 0.0
    """Probability of sampling a pure in-place turn from the current yaw rate range."""
