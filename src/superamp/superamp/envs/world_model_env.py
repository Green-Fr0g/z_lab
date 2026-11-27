"""Differentiable velocity-tracking rollout over a learned world model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from superamp.models import Dynamics
from superamp.representations import StateRepresentation
from superamp.storage import ReplayBuffer
from superamp.utils.math import quat_apply_inverse, quat_yaw

if TYPE_CHECKING:
    from superamp.amp import AMPFeatures
    from superamp.learners.discriminator_learner import DiscriminatorLearner


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


class WorldModelEnv(WorldModelRollout):
    """Train velocity control against predicted task and AMP rewards."""

    def __init__(
        self,
        dynamics: Dynamics,
        representation: StateRepresentation,
        replay_buffer: ReplayBuffer,
        amp_features: AMPFeatures,
        discriminator: DiscriminatorLearner,
        batch_size: int,
        horizon: int,
        sim_context_dim: int,
        soft_joint_pos_limits: torch.Tensor,
        reward_weights: dict[str, float],
        reward_stds: dict[str, float],
        orientation_limit: float,
        termination_softness: float = 0.05,
        *,
        history_length: int = 1,
        include_critic: bool = True,
    ) -> None:
        if termination_softness <= 0.0:
            raise ValueError("termination_softness must be positive")
        if not 1 <= history_length <= replay_buffer.buffer_size:
            raise ValueError("history_length must be in [1, buffer_size]")
        super().__init__(dynamics, representation)
        self.replay_buffer = replay_buffer
        self.amp_features = amp_features
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.horizon = horizon
        self.history_length = history_length
        self.include_critic = include_critic
        self.sim_context_dim = sim_context_dim
        self.soft_joint_pos_limits = soft_joint_pos_limits
        self.reward_weights = reward_weights
        self.reward_stds = reward_stds
        self.orientation_cosine = math.cos(orientation_limit)
        self.termination_softness = termination_softness
        amp_cfg = discriminator.cfg
        self.amp_history_length = (amp_cfg.num_frames - 1) * amp_cfg.frame_stride
        self._start_history_length = max(self.history_length, self.amp_history_length)
        self.max_step_reward = representation.dt * (
            sum(max(weight, 0.0) for weight in reward_weights.values()) + amp_cfg.reward_coef
        )

    def get_observations(self) -> TensorDict:
        """Build observations for the velocity actor and optional critic."""
        observations = {"actor": self._sensor_history(), "goal": self._goal_obs}
        if self.include_critic:
            observations["critic"] = self._critic_obs()
        return TensorDict(observations, batch_size=[self.batch_size])

    def _build_task_goals(self) -> None:
        """Select the recorded (vx, vy, yaw rate) command for the current rollout step."""
        self._goal_obs = self.goal_window[:, self.goal_step]

    @property
    def num_actions(self) -> int:
        return self.dynamics.num_actions

    @property
    def critic_input_dim(self) -> int:
        return (
            self.representation.rep_state_dim
            + 3
            + self.sim_context_dim
            + self.num_actions
            + self.amp_history_length * self.amp_features.observation_dim
        )

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
        """Advance model state and return observations, reward, and soft termination."""
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
        return torch.cat(
            (self.rep_state, self._goal_obs, self.sim_ctx, self.last_action, self.amp_history.flatten(1)), dim=-1
        )

    def _wm_step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score the predicted state against the current goal and advance AMP history."""
        next_aux, next_rep = self._predict_state(action)
        next_joint_state = self.representation.joint_state(next_rep)
        bodies = self.representation.world_frame_bodies(next_aux, next_rep)
        task_reward = self._task_reward(next_aux, next_joint_state, bodies, self._goal_obs, action)
        next_amp = self.amp_features.from_bodies(*bodies)
        amp_window = torch.cat((self.amp_history, next_amp.unsqueeze(1)), dim=1)
        amp_reward = self.discriminator.reward(amp_window[:, :: self.discriminator.cfg.frame_stride])
        reward = task_reward + self.representation.dt * amp_reward
        termination = self._compute_term_prob(next_aux)
        self.last_amp_window = amp_window
        self.amp_history = amp_window[:, 1:]
        return next_aux, next_rep, next_joint_state, reward, termination

    def _compute_term_prob(self, next_aux: torch.Tensor) -> torch.Tensor:
        up_z = 1.0 - 2.0 * next_aux[:, 3:5].square().sum(-1)
        return torch.sigmoid((self.orientation_cosine - up_z) / self.termination_softness)

    def _on_reset(self, batch: TensorDict) -> None:
        self.goal_window = batch["goal_window"]
        self.goal_step = 0
        self._update_goal()
        self.amp_history = self.amp_features.from_full_state(batch["state_window"][:, -self.amp_history_length :])
        self._latest_sensor = self._synth_sensor()
        if self.history_length > 1:
            self._past_sensors = self._warmstart_past_sensors(
                batch["sensor_window"][:, -self.history_length :], batch["pad_mask"][:, -self.history_length :]
            )

    def _compute_commands(self) -> None:
        self.goal_step += 1
        self._update_goal()

    def _update_goal(self) -> None:
        self._build_task_goals()

    def sample_initial_states(self) -> TensorDict:
        """Sample rollout starts, causal histories, and recorded velocity goals."""
        return self.replay_buffer.sample_rollout(self._start_history_length, self.horizon, self.batch_size)

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

    def _task_reward(
        self,
        aux_state: torch.Tensor,
        joint_state: torch.Tensor,
        bodies: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        goal: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        _, rotations, linear, angular = bodies
        anchor = self.amp_features.anchor_index
        lin_b = quat_apply_inverse(quat_yaw(aux_state[:, 3:7]), linear[:, anchor, :])
        lin_error = (lin_b[:, :2] - goal[:, :2]).square().sum(-1)
        ang_w = angular[:, anchor, :]
        ang_b = (rotations[:, anchor].transpose(-1, -2) @ ang_w.unsqueeze(-1)).squeeze(-1)
        ang_error = (ang_w[:, 2] - goal[:, 2]).square() + ang_b[:, :2].square().sum(-1)
        joints = joint_state[:, : self.representation.num_joints]
        lower, upper = self.soft_joint_pos_limits.unbind(-1)
        joint_error = ((lower - joints).clamp_min(0.0) + (joints - upper).clamp_min(0.0)).sum(-1)
        terms = {
            "track_lin_vel_xy": torch.exp(-lin_error / self.reward_stds["track_lin_vel_xy"] ** 2),
            "track_ang_vel_z": torch.exp(-ang_error / self.reward_stds["track_ang_vel_z"] ** 2),
            "action_rate_l2": (action - self.last_action).square().sum(-1),
            "joint_pos_limits": joint_error,
        }
        return self.representation.dt * sum(self.reward_weights[name] * value for name, value in terms.items())
