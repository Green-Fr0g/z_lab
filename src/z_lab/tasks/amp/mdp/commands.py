"""Velocity command configuration for AMP locomotion curricula."""

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform velocity commands with upper bounds for curriculum expansion."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Maximum sampling ranges allowed by the velocity curriculum."""
