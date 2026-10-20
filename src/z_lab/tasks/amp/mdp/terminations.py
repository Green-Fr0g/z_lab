# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded recovery windows for environments that would otherwise terminate immediately."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import TerminationManager

__all__ = ["DelayedTerminationManager"]


class DelayedTerminationManager(TerminationManager):
    """Adopt an existing manager and delay non-timeout terminations for selected environments.

    A counter records consecutive steps on which a non-timeout termination fires. The first
    failing step counts as one; reaching ``max_delay_steps`` releases the termination. Clearing
    the termination before that marks a recovery. Timeouts always reset immediately and never
    count as recovery successes. The fixed mask also selects recovery clips during motion resets.

    Args:
        base: Already initialized environment termination manager whose state is adopted.
        delay_env_mask: Fixed boolean mask, shape ``(num_envs,)``, on the manager's device.
        max_delay_steps: Positive number of consecutive failing control steps before reset.
    """

    def __init__(self, base: TerminationManager, delay_env_mask: torch.Tensor, max_delay_steps: int) -> None:
        if isinstance(base, DelayedTerminationManager):
            raise ValueError("A delayed termination manager is already installed.")
        if isinstance(max_delay_steps, bool) or not isinstance(max_delay_steps, int) or max_delay_steps < 1:
            raise ValueError("max_delay_steps must be a positive integer.")
        if delay_env_mask.dtype != torch.bool or delay_env_mask.shape != (base.num_envs,):
            raise ValueError(f"delay_env_mask must be a bool tensor with shape ({base.num_envs},).")
        if delay_env_mask.device != base.terminated.device:
            raise ValueError("delay_env_mask must be on the termination manager's device.")

        # Reuse parsed terms and existing buffers without registering the terms a second time.
        self.__dict__.update(base.__dict__)
        self.delay_env_mask = delay_env_mask.clone()
        self.delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
        self.delay_env_succeeded = torch.zeros_like(delay_env_mask)
        self.max_delay_steps = max_delay_steps

    def compute(self) -> torch.Tensor:
        """Compute effective dones, preserving timeouts and marking completed recovery attempts."""
        super().compute()
        timed_out = self._truncated_buf
        held = self.delay_env_mask & self._terminated_buf & ~timed_out
        recovered = self.delay_env_mask & ~self._terminated_buf & ~timed_out & (self.delay_counters > 0)

        self.delay_env_succeeded[held | timed_out] = False
        self.delay_env_succeeded[recovered] = True
        self.delay_counters[held] += 1
        self.delay_counters[recovered | timed_out] = 0
        still_held = held & (self.delay_counters < self.max_delay_steps)
        self._terminated_buf[still_held] = False
        # Expired counters remain visible to recovery rewards until the environment actually resets.
        return self.dones

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> dict[str, torch.Tensor]:
        """Reset term bookkeeping and clear recovery state only for the selected environments."""
        if env_ids is None:
            env_ids = slice(None)
        extras = super().reset(env_ids)
        self.delay_counters[env_ids] = 0
        self.delay_env_succeeded[env_ids] = False
        return extras
