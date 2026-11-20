"""Per-task observations for concurrent motion and velocity encoder training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from tensordict import TensorDict

from supermimic.algorithms.models import Dynamics, TaskSpec
from supermimic.algorithms.replay_buffer import ReplayBuffer
from supermimic.algorithms.representations import StateRepresentation
from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.tasks import VelocityTaskCfg, build_actor_obs_td, velocity_goal_from_motion


class MultiTaskWorldModelEnv(WorldModelEnv):
    """Supply each encoder with its own goal while sharing motion-tracking rewards."""

    def __init__(
        self,
        dynamics: Dynamics,
        representation: StateRepresentation,
        motion_command: Any,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        soft_joint_pos_limits: torch.Tensor,
        reward_weights: dict[str, float],
        anchor_pos_threshold: float,
        anchor_ori_threshold: float,
        termination_softness: float = 0.05,
        num_goal_frames: int = 10,
        goal_frame_stride: int = 5,
        *,
        task_specs: Sequence[TaskSpec],
        velocity_task: VelocityTaskCfg | None = None,
        history_length: int = 1,
    ) -> None:
        task_names = tuple(task.name for task in task_specs)
        if task_names not in (("motion",), ("motion", "velocity")):
            raise ValueError("task_specs must start with motion and may include velocity")
        self.task_specs = tuple(task_specs)
        self.velocity_task = velocity_task or VelocityTaskCfg()
        super().__init__(
            dynamics=dynamics,
            representation=representation,
            motion_command=motion_command,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            soft_joint_pos_limits=soft_joint_pos_limits,
            reward_weights=reward_weights,
            anchor_pos_threshold=anchor_pos_threshold,
            anchor_ori_threshold=anchor_ori_threshold,
            termination_softness=termination_softness,
            num_goal_frames=num_goal_frames,
            goal_frame_stride=goal_frame_stride,
            history_length=history_length,
        )

    def _build_task_goals(self) -> None:
        self._task_goals = {"motion": self._actor_goal_obs}
        if len(self.task_specs) > 1:
            self._task_goals["velocity"] = velocity_goal_from_motion(
                self.motion, self.motion_times, self.aux_state[:, 3:7], self.velocity_task
            )

    def get_observations(self) -> TensorDict:
        """Emit full-batch goals so each encoder can select its assigned slots."""
        return build_actor_obs_td(self._task_goals, self._sensor_history(), self._critic_obs())
