"""Collect simulator experience and train SuperAMP's world model and policy."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from tensordict import TensorDict

from superamp.envs import IsaacLabSuperAmpAdapter
from superamp.envs.isaaclab_world_model import build_world_model_from_state
from superamp.envs.world_model_env import WorldModelEnv
from superamp.learners.discriminator_learner import DiscriminatorLearner, DiscriminatorLearnerCfg
from superamp.learners.policy_learner import (
    ActorCriticPolicyLearner,
    ActorCriticPolicyLearnerCfg,
    BpttPolicyLearner,
    BpttPolicyLearnerCfg,
)
from superamp.learners.world_model_learner import WorldModelLearner, WorldModelLearnerCfg
from superamp.models import VelocityActor, build_mlp
from superamp.representations import PROPRIO_OBS_TERMS
from superamp.storage import AMPReplayBuffer, FieldSpec, ReplayBuffer
from superamp.storage.replay_buffer import NoValidSequenceError
from superamp.utils.logger import TrainingLogger


@dataclass
class SuperAmpRunnerCfg:
    """Model, policy, discriminator, and training settings."""

    representation: Literal["gencoord", "maxcoord"]
    max_iterations: int = 100_000
    log_interval: int = 10
    save_interval: int = 1000
    replay_buffer_size: int = 256
    amp_replay_buffer_size: int = 100_000
    warmup_steps: int = 16
    replay_ratio: float = 4.0
    history_length: int = 1
    action_noise: float = 0.1
    dynamics_hidden_dims: tuple[int, ...] = (1024, 1024, 1024)
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    proprio_noise: dict[str, float] = field(default_factory=lambda: dict.fromkeys(PROPRIO_OBS_TERMS, 0.0))
    world_model: WorldModelLearnerCfg = field(
        default_factory=lambda: WorldModelLearnerCfg(num_batches=2, batch_size=1024)
    )
    policy: BpttPolicyLearnerCfg = field(
        default_factory=lambda: ActorCriticPolicyLearnerCfg(horizon=8, batch_size=1024)
    )
    discriminator: DiscriminatorLearnerCfg = field(default_factory=DiscriminatorLearnerCfg)

    def __post_init__(self) -> None:
        self.dynamics_hidden_dims = tuple(self.dynamics_hidden_dims)
        self.actor_hidden_dims = tuple(self.actor_hidden_dims)
        self.critic_hidden_dims = tuple(self.critic_hidden_dims)


def build_actor(cfg: SuperAmpRunnerCfg, adapter: IsaacLabSuperAmpAdapter) -> VelocityActor:
    """Build the velocity actor used by both training and playback."""
    return VelocityActor(
        adapter.validate_proprio_layout(),
        adapter.observation_dims["goal"],
        adapter.num_actions,
        cfg.actor_hidden_dims,
        cfg.history_length,
    ).to(adapter.device)


def build_actor_observations(sensors: torch.Tensor, goal: torch.Tensor) -> TensorDict:
    return TensorDict({"actor": sensors, "goal": goal}, batch_size=[sensors.shape[0]])


def advance_proprio_history(history: torch.Tensor, sensor: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    newest = sensor.unsqueeze(1)
    return torch.where(dones[:, None, None], newest, torch.cat((history[:, 1:], newest), dim=1))


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


class SuperAmpRunner:
    """Run simulator collection, supervised dynamics, AMP, and BPTT updates."""

    def __init__(self, env: object, cfg: SuperAmpRunnerCfg) -> None:
        self.adapter = IsaacLabSuperAmpAdapter(env)
        self.env = self.adapter.env
        self.cfg = cfg
        amp_history = (cfg.discriminator.num_frames - 1) * cfg.discriminator.frame_stride
        if cfg.history_length < 1 or cfg.replay_buffer_size < max(
            cfg.world_model.horizon, cfg.history_length, amp_history
        ):
            raise ValueError("Replay capacity must cover world-model, actor, and AMP histories")
        if min(cfg.max_iterations, cfg.log_interval, cfg.save_interval, cfg.warmup_steps) < 1 or cfg.replay_ratio <= 0:
            raise ValueError("Iteration counts, warmup steps, and replay ratio must be positive")
        self._init_common()
        self._init_policy()
        self._seed_sim_history()
        samples_per_update = cfg.world_model.num_batches * cfg.world_model.batch_size * cfg.world_model.horizon
        self._updates_per_step = self.adapter.num_envs * cfg.replay_ratio / samples_per_update
        amp_cfg = cfg.discriminator
        self._amp_batch_size = self.adapter.num_envs * amp_cfg.num_steps_per_env // amp_cfg.num_mini_batches
        if self._amp_batch_size < 1:
            raise ValueError("Discriminator minibatches must contain at least one window")
        self._amp_update_credit = 0
        self._update_credit = 0.0
        self._iteration = -1
        self._logger_state: dict = {}

    def _init_common(self) -> None:
        cfg = self.cfg
        self.env.reset()
        self._wm = build_world_model_from_state(self.adapter, cfg.representation, cfg.dynamics_hidden_dims)
        self.representation = self._wm.representation
        self.representation.configure_actor_observation(self.adapter.num_actions, cfg.proprio_noise)
        self.dynamics = self._wm.dynamics
        self.world_model_learner = WorldModelLearner(self.dynamics, self.representation, cfg.world_model)
        amp_cfg = cfg.discriminator
        adapter = self.adapter
        dims = adapter.observation_dims
        device = adapter.device
        schema = {
            "state": FieldSpec(dims["full_state"]),
            "next_state": FieldSpec(dims["full_state"]),
            "next_state_valid": FieldSpec(dtype=torch.bool),
            "actor_obs": FieldSpec(dims["actor"]),
            "goal": FieldSpec(dims["goal"]),
            "action": FieldSpec(adapter.num_actions),
            "prev_action": FieldSpec(adapter.num_actions),
            "sim_ctx": FieldSpec(self._wm.sim_context_dim),
            "terminated": FieldSpec(dtype=torch.bool),
            "truncated": FieldSpec(dtype=torch.bool),
        }
        self.replay_buffer = ReplayBuffer(schema, cfg.replay_buffer_size, adapter.num_envs, device)
        self.amp_replay = AMPReplayBuffer(
            self._wm.amp_features.observation_dim,
            cfg.amp_replay_buffer_size,
            device,
            num_frames=amp_cfg.num_frames,
            frame_stride=amp_cfg.frame_stride,
        )
        self.expert = self.adapter.expert_transitions(num_frames=amp_cfg.num_frames, frame_stride=amp_cfg.frame_stride)
        self.discriminator_learner = DiscriminatorLearner(self.expert.observation_dim, amp_cfg, self.adapter.device)

    def _init_policy(self) -> None:
        cfg = self.cfg
        self.actor = build_actor(cfg, self.adapter)
        self.world_model_env = WorldModelEnv(
            dynamics=self.dynamics,
            representation=self.representation,
            replay_buffer=self.replay_buffer,
            amp_features=self._wm.amp_features,
            discriminator=self.discriminator_learner,
            batch_size=cfg.policy.batch_size,
            horizon=cfg.policy.horizon,
            sim_context_dim=self._wm.sim_context_dim,
            soft_joint_pos_limits=self.adapter.soft_joint_pos_limits(),
            reward_weights=self.adapter.reward_weights(),
            reward_params=self.adapter.reward_params(),
            orientation_limit=self.env.cfg.terminations.bad_orientation.params["limit_angle"],
            history_length=cfg.history_length,
        )
        self.policy_learner: BpttPolicyLearner
        if isinstance(cfg.policy, ActorCriticPolicyLearnerCfg):
            critic = build_mlp(self.world_model_env.critic_input_dim, cfg.critic_hidden_dims, 1).to(self.adapter.device)
            self.policy_learner = ActorCriticPolicyLearner(self.actor, critic, self.world_model_env, cfg.policy)
        else:
            self.policy_learner = BpttPolicyLearner(self.actor, self.world_model_env, cfg.policy)

    def _seed_sim_history(self) -> None:
        observations = self.adapter.observations()
        self._init_sim_amp_history(observations["full_state"])
        sensor = self.representation.add_proprio_noise(observations["actor"])
        self._sim_history = sensor.unsqueeze(1).repeat(1, self.cfg.history_length, 1)

    def _init_sim_amp_history(self, full_state: torch.Tensor) -> None:
        adapter = self.adapter
        device = adapter.device
        # The newest frame is supplied by the current transition, so the ring stores only its history.
        self._amp_history_length = self.amp_replay.window_length - 1
        self._amp_history = torch.empty(
            adapter.num_envs, self._amp_history_length, self._wm.amp_features.observation_dim, device=device
        )
        self._amp_history_count = torch.zeros(adapter.num_envs, dtype=torch.long, device=device)
        self._amp_history_slots = torch.arange(1, self._amp_history_length + 1, device=device)
        self._amp_history_slot = 0
        self._current_amp = self._wm.amp_features.from_full_state(full_state)

    @torch.no_grad()
    def step(self, logger: TrainingLogger | None = None) -> torch.Tensor:
        """Collect a simulator step using actions from the current velocity actor."""
        obs = self.adapter.observations()
        actor_obs = build_actor_observations(self._sim_history, obs["goal"].clone())
        self.actor.update_normalization(actor_obs)
        actions = self.actor(actor_obs) + torch.randn_like(self.env.action_manager.action) * self.cfg.action_noise
        state = obs["full_state"]
        fields = {
            "state": obs["full_state"],
            "actor_obs": obs["actor"].clone(),
            "goal": actor_obs["goal"],
            "action": actions,
            "prev_action": self.adapter.previous_actions(),
            "sim_ctx": self.adapter.sim_context(),
        }
        next_obs, rewards, terminated, truncated, extras = self.adapter.step(actions)

        dones = terminated | truncated
        # Isaac Lab's reset_buf covers automatic resets. GUI resets can also set terminated,
        # but are handled after final_obs was collected and have no valid final observation.
        autoreset = self.env.reset_buf
        next_state_valid = ~dones | autoreset
        next_state = next_obs["full_state"].clone()
        rows = autoreset.nonzero(as_tuple=True)[0]
        if rows.numel():
            final = extras["final_obs"]
            next_state[rows] = final["full_state"][rows]
        next_state = torch.where(next_state_valid[:, None], next_state, fields["state"])
        fields.update(
            next_state=next_state, next_state_valid=next_state_valid, terminated=terminated, truncated=truncated
        )
        self.replay_buffer.insert(TensorDict(fields, batch_size=[self.adapter.num_envs]))
        self._record_sim_amp(next_obs["full_state"], next_state, rows, next_state_valid, dones)
        self.world_model_learner.update_normalization(state, actions)
        sensor = self.representation.add_proprio_noise(next_obs["actor"])
        self._sim_history = advance_proprio_history(self._sim_history, sensor, dones)
        if logger is not None:
            logger.process_env_step(rewards, dones, extras)
        return rewards

    def _record_sim_amp(
        self,
        observed_state: torch.Tensor,
        next_state: torch.Tensor,
        reset_rows: torch.Tensor,
        next_state_valid: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self._amp_history[:, self._amp_history_slot].copy_(self._current_amp)
        self._current_amp = self._wm.amp_features.from_full_state(observed_state)
        next_amp = self._current_amp.clone()
        if reset_rows.numel():
            next_amp[reset_rows] = self._wm.amp_features.from_full_state(next_state[reset_rows])
        self._store_amp_windows(next_amp, next_state_valid, dones)

    def _store_amp_windows(self, next_amp: torch.Tensor, next_state_valid: torch.Tensor, ended: torch.Tensor) -> None:
        self._amp_history_count.add_(1).clamp_max_(self._amp_history_length)
        ready = (self._amp_history_count == self._amp_history_length) & next_state_valid
        rows = ready.nonzero(as_tuple=True)[0]
        slots = (self._amp_history_slot + self._amp_history_slots) % self._amp_history_length
        windows = torch.cat((self._amp_history[rows[:, None], slots], next_amp[rows, None]), dim=1)
        self.amp_replay.insert(windows)
        self._amp_history_slot = (self._amp_history_slot + 1) % self._amp_history_length
        self._amp_history_count.masked_fill_(ended, 0)

    def _synchronize_cuda(self) -> None:
        if self.adapter.device.type == "cuda":
            torch.cuda.synchronize(self.adapter.device)

    def _train_updates(self, logger: TrainingLogger) -> tuple[float, float]:
        cfg = self.cfg
        if self.replay_buffer.num_transitions < min(
            max(cfg.warmup_steps, cfg.world_model.horizon), cfg.replay_buffer_size
        ):
            return 0.0, 0.0
        # Spread the PPO discriminator budget over model updates as simulator steps arrive.
        self._amp_update_credit += cfg.discriminator.num_learning_epochs * cfg.discriminator.num_mini_batches
        self._update_credit += self._updates_per_step
        updates = int(self._update_credit)
        self._update_credit -= updates
        world_model_time = policy_time = 0.0
        for _ in range(updates):
            start = time.perf_counter()
            try:
                sequences = [
                    self.replay_buffer.sample(cfg.world_model.horizon, cfg.world_model.batch_size)
                    for _ in range(cfg.world_model.num_batches)
                ]
            except NoValidSequenceError:
                break
            logger.process_update(self.world_model_learner.update(sequences))
            self._synchronize_cuda()
            policy_start = time.perf_counter()
            world_model_time += policy_start - start
            logger.process_update(self._update_policy())
            self._synchronize_cuda()
            policy_time += time.perf_counter() - policy_start
        return world_model_time, policy_time

    def _update_policy(self) -> dict[str, torch.Tensor]:
        if not self.amp_replay.size:
            return {}
        num_batches, self._amp_update_credit = divmod(self._amp_update_credit, self.cfg.discriminator.num_steps_per_env)
        losses = {}
        if num_batches:
            losses = self.discriminator_learner.update(self.expert, self.amp_replay, self._amp_batch_size, num_batches)
        losses.update(self.policy_learner.update())
        return losses

    def run(self, log_dir: Path, infos: dict | None = None) -> Path:
        """Train and periodically save complete and standalone world-model checkpoints."""
        from torch.utils.tensorboard import SummaryWriter

        log_dir.mkdir(parents=True, exist_ok=True)
        start_iteration = self._iteration + 1
        end_iteration = start_iteration + self.cfg.max_iterations
        collect_time = world_model_time = policy_time = 0.0
        collected_steps = 0
        with SummaryWriter(log_dir=str(log_dir), flush_secs=10) as writer:
            logger = TrainingLogger(
                writer,
                log_dir.name,
                (infos or {}).get("task"),
                self.adapter.num_envs,
                self.adapter.device,
                start_iteration,
                self._logger_state,
            )
            for offset in range(self.cfg.max_iterations):
                start = time.perf_counter()
                self.step(logger)
                self._synchronize_cuda()
                collected_steps += self.adapter.num_envs
                collect_time += time.perf_counter() - start
                model_time, update_time = self._train_updates(logger)
                world_model_time += model_time
                policy_time += update_time
                self._iteration = start_iteration + offset
                final = offset + 1 == self.cfg.max_iterations
                if (offset + 1) % self.cfg.log_interval == 0 or final:
                    logger.log(
                        self._iteration,
                        end_iteration,
                        collect_time,
                        world_model_time,
                        policy_time,
                        collected_steps,
                    )
                    collect_time = world_model_time = policy_time = 0.0
                    collected_steps = 0
                if (offset + 1) % self.cfg.save_interval == 0 or final:
                    self._logger_state = logger.state_dict()
                    self.save(log_dir / f"model_{self._iteration}.pt", infos)
                    _save_checkpoint(log_dir / "world_model.pt", self._world_model_checkpoint())
        return log_dir

    def _world_model_checkpoint(self) -> dict:
        return {
            "format_version": 1,
            "iter": self._iteration,
            "representation": self.cfg.representation,
            "joint_names": list(self.adapter.action_term._joint_names),
            "body_names": list(self.env.cfg.full_state_body_names),
            "config": {"dynamics_hidden_dims": self.cfg.dynamics_hidden_dims},
            "dynamics": self.dynamics.state_dict(),
        }

    def save(self, path: Path, infos: dict | None = None) -> None:
        """Save networks, normalizers, optimizer states and configuration."""
        learner = self.discriminator_learner
        checkpoint = {
            **self._world_model_checkpoint(),
            "infos": infos,
            "logger": self._logger_state,
            "config": asdict(self.cfg),
            "amp_body_names": list(self.env.cfg.amp_body_names),
            "amp_anchor_name": self.env.cfg.anchor_link_name,
            "amp_linear_velocity_at_com": False,
            "actor": self.actor.state_dict(),
            "actor_optimizers": [optimizer.state_dict() for optimizer in self.policy_learner.actor_optimizers],
            "world_model_optimizers": [optimizer.state_dict() for optimizer in self.world_model_learner.optimizers],
            "discriminator": learner.discriminator.state_dict(),
            "discriminator_normalizer": learner.normalizer.state_dict(),
            "discriminator_optimizer": learner.optimizer.state_dict(),
        }
        if isinstance(self.policy_learner, ActorCriticPolicyLearner):
            checkpoint["critic"] = self.policy_learner.critic.state_dict()
            checkpoint["critic_optimizers"] = [
                optimizer.state_dict() for optimizer in self.policy_learner.critic_optimizers
            ]
        _save_checkpoint(path, checkpoint)

    def load_world_model(self, checkpoint: dict) -> None:
        """Initialize dynamics and its normalizers from a world-model or full checkpoint."""
        layout = self._world_model_checkpoint()
        for name in ("format_version", "representation", "joint_names", "body_names"):
            if layout[name] != checkpoint[name]:
                raise ValueError(f"Checkpoint {name} must match the SuperAMP task")
        if tuple(checkpoint["config"]["dynamics_hidden_dims"]) != self.cfg.dynamics_hidden_dims:
            raise ValueError("Checkpoint dynamics dimensions must match the agent")
        state = checkpoint["dynamics"]
        for name in ("action_scale", "action_offset", "representation.default_joint_pos"):
            target = layout["dynamics"][name]
            if not torch.equal(state[name].to(target), target):
                raise ValueError(f"Checkpoint dynamics {name} must match the training task")
        self.dynamics.load_state_dict(state)

    def load(self, checkpoint: dict) -> None:
        """Restore training state and collect fresh simulator replay."""
        cfg = checkpoint["config"]
        if self.cfg.history_length != cfg["history_length"] or self.cfg.actor_hidden_dims != tuple(
            cfg["actor_hidden_dims"]
        ):
            raise ValueError("Checkpoint actor layout must match the agent")
        for name in ("num_frames", "frame_stride"):
            if getattr(self.cfg.discriminator, name) != cfg["discriminator"][name]:
                raise ValueError(f"Checkpoint discriminator {name} must match the agent")
        if (
            checkpoint["amp_body_names"] != list(self.env.cfg.amp_body_names)
            or checkpoint["amp_anchor_name"] != self.env.cfg.anchor_link_name
            or checkpoint["amp_linear_velocity_at_com"]
        ):
            raise ValueError("Checkpoint AMP feature layout must match the task")
        policy = self.policy_learner
        if isinstance(policy, ActorCriticPolicyLearner) != ("critic" in checkpoint):
            raise ValueError("Checkpoint policy learner must match the agent")

        self.load_world_model(checkpoint)
        self.actor.load_state_dict(checkpoint["actor"])
        for optimizer, state in zip(policy.actor_optimizers, checkpoint["actor_optimizers"], strict=True):
            optimizer.load_state_dict(state)
        for optimizer, state in zip(
            self.world_model_learner.optimizers, checkpoint["world_model_optimizers"], strict=True
        ):
            optimizer.load_state_dict(state)
        learner = self.discriminator_learner
        learner.discriminator.load_state_dict(checkpoint["discriminator"])
        learner.normalizer.load_state_dict(checkpoint["discriminator_normalizer"])
        learner.optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        if isinstance(policy, ActorCriticPolicyLearner):
            policy.critic.load_state_dict(checkpoint["critic"])
            for optimizer, state in zip(policy.critic_optimizers, checkpoint["critic_optimizers"], strict=True):
                optimizer.load_state_dict(state)
        self._logger_state = checkpoint.get("logger", {})
        self._iteration = checkpoint["iter"]
