"""Differentiable motion-tracking rollout over a learned world model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from tensordict import TensorDict

from supermimic.models import Dynamics
from supermimic.representations import StateRepresentation
from supermimic.storage import ReplayBuffer


class WorldModelRollout:
    """Own the learned transition state shared by training and playback."""

    def __init__(self, dynamics: Dynamics, representation: StateRepresentation) -> None:
        self.dynamics = dynamics
        self.representation = representation

    def reset_dynamics(
        self,
        full_state: torch.Tensor,
        sim_ctx: torch.Tensor,
        last_action: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Initialize from simulation, optionally replacing only the specified rows."""
        if env_ids is None:
            self.aux_state = self.representation.extract_aux_state(full_state).detach().clone()
            self.rep_state = self.representation.represent(full_state).detach().clone()
            self.sim_ctx = sim_ctx.detach().clone()
            self.last_action = last_action.detach().clone()
        else:
            state = full_state[env_ids]
            self.aux_state[env_ids] = self.representation.extract_aux_state(state).detach()
            self.rep_state[env_ids] = self.representation.represent(state).detach()
            self.sim_ctx[env_ids] = sim_ctx[env_ids].detach()
            self.last_action[env_ids] = last_action[env_ids].detach()

    def _predict_state(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dynamics(self.rep_state, action, self.aux_state, self.sim_ctx)

    def advance_dynamics(self, action: torch.Tensor) -> None:
        """Advance the learned state by one simulator control step."""
        self.aux_state, self.rep_state = self._predict_state(action)
        self.last_action = action.detach().clone()


class WorldModelEnv(WorldModelRollout, ABC):
    """Advance a learned robot state against a shared reference motion."""

    def __init__(
        self,
        dynamics: Dynamics,
        representation: StateRepresentation,
        motion_command: Any,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        soft_joint_pos_limits: torch.Tensor,
        reward_weights: dict[str, float],
        reward_stds: dict[str, float],
        anchor_pos_threshold: float,
        anchor_ori_threshold: float,
        termination_softness: float = 0.05,
        num_goal_frames: int = 10,
        goal_frame_stride: int = 5,
        *,
        history_length: int = 1,
    ) -> None:
        if termination_softness <= 0.0:
            raise ValueError("termination_softness must be positive")
        if not 1 <= history_length <= replay_buffer.buffer_size:
            raise ValueError("history_length must be in [1, buffer_size]")
        super().__init__(dynamics, representation)
        representation.bind(soft_joint_pos_limits=soft_joint_pos_limits)
        self.motion = motion_command.motion
        self.motion_joint_indices = motion_command.joint_indexes
        self._anchor_index = getattr(motion_command, "motion_anchor_body_index", 0)
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.history_length = history_length
        self.soft_joint_pos_limits = soft_joint_pos_limits
        self.reward_weights = reward_weights
        self.reward_stds = reward_stds
        self.anchor_pos_threshold = anchor_pos_threshold
        self.anchor_ori_threshold = anchor_ori_threshold
        self.termination_softness = termination_softness
        self.num_goal_frames = num_goal_frames
        self.goal_frame_stride = goal_frame_stride
        self.max_step_reward = representation.dt * sum(max(weight, 0.0) for weight in reward_weights.values())

    @abstractmethod
    def get_observations(self) -> TensorDict:
        """Build actor and privileged critic observations for the current state."""

    def _build_task_goals(self) -> None:  # noqa: B027
        """Refresh any additional task goals after updating the shared reference."""

    @property
    def num_actions(self) -> int:
        return self.dynamics.num_actions

    def reset(self, initial_states: TensorDict | None = None) -> TensorDict:
        """Initialize a rollout from replay or a caller-provided batch."""
        batch = self.sample_initial_states() if initial_states is None else initial_states
        self.aux_state, self.rep_state = self._unpack_initial_states(batch)
        self.joint_state = self.representation.joint_state(self.rep_state)
        self.last_action = batch["prev_action"].detach()
        self.sim_ctx = batch["sim_ctx"]
        self._on_reset(batch)
        return self.get_observations()

    def step(self, action: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor]:
        """Advance model state and return observation, reward and soft termination."""
        self.aux_state, self.rep_state, self.joint_state, reward, termination = self._wm_step(action)
        self.last_action = action
        self._compute_commands()
        if self.history_length > 1:
            self._past_sensors = torch.cat((self._past_sensors[:, 1:], self._latest_sensor.unsqueeze(1)), dim=1)
        self._latest_sensor = self._synth_sensor()
        return self.get_observations(), reward, termination

    def _synth_sensor(self) -> torch.Tensor:
        return self.representation.compute_actor_sensor(
            self.rep_state,
            self.sim_ctx,
            aux_state=self.aux_state,
            joint_state=self.joint_state,
            last_action=self.last_action,
            noisy=True,
        )

    def _sensor_history(self) -> torch.Tensor:
        latest = self._latest_sensor.unsqueeze(1)
        if self.history_length == 1:
            return latest
        return torch.cat((self._past_sensors, latest), dim=1)

    def _critic_obs(self) -> torch.Tensor:
        return torch.cat((self.rep_state, self._wm_goal_obs, self.sim_ctx), dim=-1)

    def _wm_step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score the predicted state against the reference before advancing its frame."""
        next_aux, next_rep = self._predict_state(action)
        next_joint_state = self.representation.joint_state(next_rep)
        reward = self.representation.compute_wm_reward_from_reference(
            next_rep,
            next_aux,
            next_joint_state,
            action,
            self.last_action,
            *self._ref_body_state,
            reward_weights=self.reward_weights,
            reward_stds=self.reward_stds,
        )
        termination = self._compute_term_prob(next_aux)
        return next_aux, next_rep, next_joint_state, reward, termination

    def _compute_term_prob(self, next_aux: torch.Tensor) -> torch.Tensor:
        target_pos, target_quat, _, _ = self._ref_body_state
        height_error = (target_pos[:, self._anchor_index, 2] - next_aux[:, 2]).abs()
        target_gravity_z = 2 * target_quat[:, self._anchor_index, :2].square().sum(-1) - 1
        current_quat = next_aux[:, 3:7]
        current_gravity_z = 2 * current_quat[:, :2].square().sum(-1) - 1
        tilt_error = (target_gravity_z - current_gravity_z).abs()
        normalized_error = torch.maximum(
            height_error / self.anchor_pos_threshold,
            tilt_error / self.anchor_ori_threshold,
        )
        return torch.sigmoid((normalized_error - 1.0) / self.termination_softness)

    def _on_reset(self, batch: TensorDict) -> None:
        self.motion_times = batch["motion_time"].to(torch.long)
        self._update_goal()
        self._latest_sensor = self._synth_sensor()
        if self.history_length > 1:
            self._past_sensors = self._warmstart_past_sensors(batch["sensor_window"], batch["pad_mask"])

    def _compute_commands(self) -> None:
        clip_end = self.motion.clip_end(self.motion_times)
        self.motion_times = torch.minimum(self.motion_times + 1, clip_end - 1)
        self._update_goal()

    def _update_goal(self) -> None:
        self._ref_body_state = self.motion.body_state(self.motion_times)
        self._wm_goal_obs, self._actor_goal_obs = self.representation.compute_wm_goal(
            self.motion,
            self.motion_times,
            self.aux_state,
            self.num_goal_frames,
            self.goal_frame_stride,
            self.motion_joint_indices,
        )
        self._build_task_goals()

    def sample_initial_states(self) -> TensorDict:
        """Sample current transitions and, when enabled, their causal histories."""
        if self.history_length == 1:
            return self.replay_buffer.sample(sequence_length=1, batch_size=self.batch_size)[:, 0]
        window, padded = self.replay_buffer.sample_causal(self.history_length, self.batch_size)
        batch = window[:, -1]
        batch["sensor_window"] = window["actor_obs"]
        batch["pad_mask"] = padded
        return batch

    def _warmstart_past_sensors(self, sensor_window: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        past = sensor_window[:, :-1].detach()
        past = self.representation.add_proprio_noise(past)
        history = torch.cat((past, self._latest_sensor.unsqueeze(1)), dim=1)
        first_real = (~pad_mask).int().argmax(dim=1)
        oldest = history[torch.arange(history.shape[0], device=history.device), first_real]
        return torch.where(pad_mask[:, :-1, None], oldest[:, None], past)

    def _unpack_initial_states(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        state = batch["state"]
        return self.representation.extract_aux_state(state).detach(), self.representation.represent(state).detach()
