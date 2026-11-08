"""Velocity command curricula for AMP locomotion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def recovery_assist_force(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_name: str = "recovery_assist",
) -> float:
    """Log the assistance magnitude in N, updated every control step by its event term."""
    return env.event_manager.get_term_cfg(event_name).func.current_force


class commands_vel(ManagerTermBase):
    """Linearly expand velocity ranges to their limits over global control steps.

    The initial ranges are captured once when the term is constructed. Each reset
    interpolates both endpoints from those initial ranges to ``limit_ranges``.
    Commands are sampled uniformly within the resulting ranges by the command term;
    existing commands keep their values until resampling.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if cfg.params["num_steps"] <= 0:
            raise ValueError("Velocity curriculum num_steps must be positive.")
        command_term = env.command_manager.get_term(cfg.params["command_name"])
        self._initial_ranges = command_term.cfg.ranges.copy()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        num_steps: int,
    ) -> dict[str, torch.Tensor]:
        del env_ids
        command_term = env.command_manager.get_term(command_name)
        ranges = command_term.cfg.ranges
        progress = min(max(env.common_step_counter / num_steps, 0.0), 1.0)
        state = {"progress": torch.tensor(progress, device=env.device)}
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            initial = getattr(self._initial_ranges, name)
            limits = getattr(command_term.cfg.limit_ranges, name)
            value = tuple(start + progress * (end - start) for start, end in zip(initial, limits, strict=True))
            setattr(ranges, name, value)
            state[f"{name}_min"] = torch.tensor(value[0], device=env.device)
            state[f"{name}_max"] = torch.tensor(value[1], device=env.device)
        return state
