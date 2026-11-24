"""Episode boundaries for SuperMimic tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedEnv

    from .commands import MotionCommand


def end_of_motion(env: ManagerBasedEnv) -> torch.Tensor:
    """Truncate after scoring the last frame of the reference motion."""
    command: MotionCommand = env.command_manager.get_term("motion")
    return command.time_steps + 1 == command.motion.clip_end(command.time_steps)
