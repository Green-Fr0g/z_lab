"""Assemble and train SuperMimic on a manager-based Isaac Lab environment."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from tensordict import TensorDict

from supermimic.envs.isaaclab_adapter import IsaacLabSuperMimicAdapter
from supermimic.envs.isaaclab_world_model import WorldModel, build_world_model_from_state
from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.learners.policy_learner import (
    ActorCriticPolicyLearner,
    ActorCriticPolicyLearnerCfg,
    BpttPolicyLearner,
    BpttPolicyLearnerCfg,
)
from supermimic.learners.world_model_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import MultiEncoderAE, TaskSpec, build_mlp
from supermimic.representations import extract_actor_goal
from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
from supermimic.storage import FieldSpec, ReplayBuffer
from supermimic.tasks import VelocityTaskCfg, advance_proprio_history, build_actor_obs_td, velocity_goal_from_motion
from supermimic.utils.logger import TrainingLogger


@dataclass
class BaseRunnerCfg:
    """Shared world model, replay and observation settings."""

    replay_buffer_size: int
    representation: Literal["gencoord", "maxcoord"]
    history_length: int
    dynamics_hidden_dims: tuple[int, ...]
    world_model: WorldModelLearnerCfg
    proprio_noise: dict[str, float]
    log_interval: int
    save_interval: int


@dataclass
class SuperMimicRunnerCfg(BaseRunnerCfg):
    """Multi-encoder policy, training and reset sampling settings."""

    max_iterations: int
    replay_ratio: float
    action_noise: float
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    latent_dim: int
    latent_bc_weight: float
    slot_mix: dict[str, float]
    velocity: VelocityTaskCfg
    sampling_mode: Literal["return", "uniform", "start", "adaptive"]
    policy: BpttPolicyLearnerCfg | ActorCriticPolicyLearnerCfg
    return_sampler: ReturnSamplerCfg


def build_world_model(cfg: BaseRunnerCfg, adapter: IsaacLabSuperMimicAdapter) -> WorldModel:
    """Reset the task and build its representation and dynamics."""
    adapter.env.reset()
    return build_world_model_from_state(adapter, cfg.representation, cfg.dynamics_hidden_dims)


def build_actor(
    cfg: SuperMimicRunnerCfg, world_model: WorldModel, num_goal_frames: int, device: torch.device
) -> MultiEncoderAE:
    """Build the actor with Motion and Velocity goal encoders."""
    num_actions = world_model.dynamics.num_actions
    motion_goal_dim = world_model.num_goals - 3 * num_goal_frames
    if motion_goal_dim != num_goal_frames * (2 * num_actions + 6):
        raise ValueError("goal observation does not match the configured joint-motion window")
    if any(cfg.slot_mix.get(name, 0.0) <= 0.0 for name in ("motion", "velocity")):
        raise ValueError("motion and velocity slot fractions must both be positive")
    return MultiEncoderAE(
        obs_dim=world_model.proprio_dim,
        action_dim=num_actions,
        task_specs=[TaskSpec("motion", motion_goal_dim), TaskSpec("velocity", 3)],
        slot_mix=cfg.slot_mix,
        latent_dim=cfg.latent_dim,
        hidden_dims=cfg.actor_hidden_dims,
        history_length=cfg.history_length,
        latent_bc_weight=cfg.latent_bc_weight,
    ).to(device)


class Ratio:
    """Carry fractional model updates across environment collection steps."""

    def __init__(self, ratio: float) -> None:
        self._ratio = ratio
        self._previous_step: float | None = None

    def __call__(self, step: int) -> int:
        if self._ratio == 0:
            return 0
        if self._ratio < 0:
            return 1
        if self._previous_step is None:
            self._previous_step = step
            return 1
        updates = int((step - self._previous_step) * self._ratio)
        self._previous_step += updates / self._ratio
        return updates


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


class BaseRunner[CfgT: BaseRunnerCfg](ABC):
    """Share world-model setup, replay and simulator collection across runners."""

    def __init__(self, env: object, cfg: CfgT) -> None:
        self.adapter = IsaacLabSuperMimicAdapter(env)
        self.env = self.adapter.env
        self.cfg = cfg
        self._init_common()
        self._init_policy()
        self._seed_sim_history()

    def _init_common(self) -> None:
        self._wm = build_world_model(self.cfg, self.adapter)
        self.representation = self._wm.representation
        self.representation.configure_actor_observation(self.adapter.num_actions, self.cfg.proprio_noise)
        self.dynamics = self._wm.dynamics
        self.replay_buffer = ReplayBuffer(
            self._buffer_schema(), self.cfg.replay_buffer_size, self.adapter.num_envs, self.adapter.device
        )
        self.world_model_learner = WorldModelLearner(self.dynamics, self.representation, self.cfg.world_model)

    def _seed_sim_history(self) -> None:
        sensor = self.representation.add_proprio_noise(self.adapter.observations()["actor"])
        self._sim_history = sensor.unsqueeze(1).repeat(1, self.cfg.history_length, 1)

    def _advance_sim_history(self, next_actor: torch.Tensor, dones: torch.Tensor) -> None:
        sensor = self.representation.add_proprio_noise(next_actor)
        self._sim_history = advance_proprio_history(self._sim_history, sensor, dones)

    @property
    def _stores_actor_obs(self) -> bool:
        return self.cfg.history_length > 1

    def _buffer_schema(self) -> dict[str, FieldSpec]:
        schema = {
            "state": FieldSpec(self.representation.full_state_dim),
            "action": FieldSpec(self.adapter.num_actions),
            "prev_action": FieldSpec(self.adapter.num_actions),
            "motion_time": FieldSpec(dtype=torch.long),
            "terminated": FieldSpec(dtype=torch.bool),
            "truncated": FieldSpec(dtype=torch.bool),
            "sim_ctx": FieldSpec(self._wm.sim_context_dim),
        }
        if self._stores_actor_obs:
            schema["actor_obs"] = FieldSpec(self._wm.proprio_dim)
        return schema

    @abstractmethod
    def _init_policy(self) -> None:
        """Build the policy and its learner for this runner."""

    def _step_and_store(
        self,
        actions: torch.Tensor,
        observations: dict[str, torch.Tensor],
        motion_time: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the simulator and attach its outcome to the pre-step data."""
        fields = {
            "state": observations["full_state"],
            "action": actions,
            "prev_action": self.adapter.previous_actions(),
            "motion_time": motion_time,
            "sim_ctx": sim_ctx,
        }
        if self._stores_actor_obs:
            fields["actor_obs"] = observations["actor"].clone()
        next_observations, rewards, terminated, truncated, extras = self.adapter.step(actions)
        fields["terminated"] = terminated.bool()
        fields["truncated"] = truncated.bool()
        self.replay_buffer.insert(TensorDict(fields, batch_size=[self.adapter.num_envs]))
        self.world_model_learner.update_normalization(fields["state"], actions)
        return next_observations, rewards, terminated, truncated, extras


class SuperMimicRunner(BaseRunner[SuperMimicRunnerCfg]):
    """Train a shared actor with Motion and Velocity encoders in model rollouts."""

    def _init_policy(self) -> None:
        cfg = self.cfg
        self.num_goal_frames, self.goal_frame_stride = self.adapter.motion_goal_window()
        self.actor = build_actor(cfg, self._wm, self.num_goal_frames, self.adapter.device)
        for batch_name, batch_size in (("simulator", self.adapter.num_envs), ("policy", cfg.policy.batch_size)):
            for task_name, (start, stop) in zip(self.actor.task_names, self.actor.slot_ranges(batch_size), strict=True):
                if start == stop:
                    raise ValueError(
                        f"slot_mix={cfg.slot_mix} assigns no slots to {task_name!r} "
                        f"in the {batch_name} batch of size {batch_size}"
                    )
        thresholds = self.env.cfg.terminations
        self.world_model_env = MultiTaskWorldModelEnv(
            dynamics=self.dynamics,
            representation=self.representation,
            motion_command=self.adapter.motion_command,
            replay_buffer=self.replay_buffer,
            batch_size=cfg.policy.batch_size,
            soft_joint_pos_limits=self.adapter.soft_joint_pos_limits(),
            reward_weights=self.adapter.reward_weights(),
            reward_stds=self.adapter.reward_stds(),
            anchor_pos_threshold=thresholds.anchor_pos.params["threshold"],
            anchor_ori_threshold=thresholds.anchor_ori.params["threshold"],
            num_goal_frames=self.num_goal_frames,
            goal_frame_stride=self.goal_frame_stride,
            task_specs=self.actor.task_specs,
            velocity_task=cfg.velocity,
            history_length=self.actor.history_length,
        )
        if self.adapter.device.type == "cuda":
            self.world_model_env.reset = torch.compile(self.world_model_env.reset)
            self.world_model_env.step = torch.compile(self.world_model_env.step)
        self.policy_learner: BpttPolicyLearner
        if isinstance(cfg.policy, ActorCriticPolicyLearnerCfg):
            critic_input_dim = self.representation.rep_state_dim + self._wm.num_goals + self._wm.sim_context_dim
            critic = build_mlp(critic_input_dim, cfg.critic_hidden_dims, 1).to(self.adapter.device)
            self.policy_learner = ActorCriticPolicyLearner(self.actor, critic, self.world_model_env, cfg.policy)
        else:
            self.policy_learner = BpttPolicyLearner(self.actor, self.world_model_env, cfg.policy)
        self.return_sampler = self._configure_reset_sampling()
        model_batch_count = cfg.world_model.num_batches * cfg.world_model.batch_size * cfg.world_model.horizon
        update_ratio = self.adapter.num_envs * cfg.replay_ratio / model_batch_count
        self._updates_per_step = Ratio(update_ratio)
        self._iteration = -1
        self._logger_state: dict = {}

    def _build_sim_obs_td(
        self,
        state: torch.Tensor,
        full_goal: torch.Tensor,
        motion_times: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> TensorDict:
        actor_goal = extract_actor_goal(full_goal, self.num_goal_frames)
        root_quat_xyzw = self.representation.extract_aux_state(state)[:, 3:7]
        critic_observation = torch.cat((self.representation.represent(state), full_goal, sim_ctx), dim=-1)
        return build_actor_obs_td(
            {
                "motion": actor_goal,
                "velocity": velocity_goal_from_motion(
                    self.adapter.motion_command.motion,
                    motion_times,
                    root_quat_xyzw,
                    self.cfg.velocity,
                ),
            },
            self._sim_history,
            critic_observation,
        )

    def _configure_reset_sampling(self) -> ReturnSampler | None:
        expected = "external" if self.cfg.sampling_mode == "return" else self.cfg.sampling_mode
        motion_command = self.adapter.motion_command
        if motion_command.cfg.sampling_mode != expected:
            raise ValueError(f"{self.cfg.sampling_mode} sampling requires Task sampling_mode={expected!r}")
        if self.cfg.sampling_mode != "return":
            return None
        return ReturnSampler(
            self.cfg.return_sampler,
            motion_command.bin_count,
            motion_command.motion.time_step_total,
            self.adapter.num_envs,
            self.world_model_env.max_step_reward,
            self.adapter.device,
        )

    @torch.no_grad()
    def step(self, logger: TrainingLogger | None = None) -> torch.Tensor:
        """Collect one simulator transition from the multi-encoder policy."""
        observations = self.adapter.observations()
        motion_time = self.adapter.motion_times()
        sim_ctx = self.adapter.sim_context()
        actor_observations = self._build_sim_obs_td(
            observations["full_state"], observations["goal"], motion_time, sim_ctx
        )
        self.actor.eval()
        actions = self.actor(actor_observations)
        actions = actions + torch.randn_like(actions) * self.cfg.action_noise
        next_observations, rewards, terminated, truncated, extras = self._step_and_store(
            actions, observations, motion_time, sim_ctx
        )
        dones = terminated | truncated
        self._advance_sim_history(next_observations["actor"], dones)
        self.actor.train()
        self.actor.update_normalization(actor_observations)
        self._update_return_sampler(motion_time, rewards, terminated, truncated)
        if logger is not None:
            logger.process_env_step(rewards, dones, extras)
        return rewards

    def _synchronize_cuda(self) -> None:
        """Wait for GPU work before reading a phase's wall-clock time."""
        if self.adapter.device.type == "cuda":
            torch.cuda.synchronize(self.adapter.device)

    def _train_updates(self, iteration: int, logger: TrainingLogger) -> tuple[float, float]:
        updates = self._updates_per_step(iteration)
        min_sequence = max(self.cfg.world_model.horizon, 2)
        if self.replay_buffer.num_transitions < min_sequence:
            return 0.0, 0.0
        world_model_time = policy_time = 0.0
        for _ in range(updates):
            start = time.perf_counter()
            sequences = [
                self.replay_buffer.sample(self.cfg.world_model.horizon, self.cfg.world_model.batch_size)
                for _ in range(self.cfg.world_model.num_batches)
            ]
            logger.process_update(self.world_model_learner.update(sequences))
            self._synchronize_cuda()
            policy_start = time.perf_counter()
            world_model_time += policy_start - start
            logger.process_update(self.policy_learner.update())
            self._synchronize_cuda()
            policy_time += time.perf_counter() - policy_start
        return world_model_time, policy_time

    def _update_return_sampler(
        self,
        motion_time: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        if self.return_sampler is None:
            return
        termination_manager = self.env.termination_manager
        clip_end = termination_manager.get_term("motion_end").bool()
        timeout = truncated.bool() & ~clip_end
        weights = self.return_sampler.record(
            motion_time,
            rewards,
            terminated.bool() & ~truncated.bool(),
            clip_end,
            timeout,
        )
        if weights is not None:
            self.adapter.motion_command.set_sampling_distribution(weights)

    def run(self, log_dir: Path, infos: dict | None = None) -> Path:
        """Collect transitions, update both learners and save training checkpoints."""
        from torch.utils.tensorboard import SummaryWriter

        log_dir.mkdir(parents=True, exist_ok=True)
        start_iteration = self._iteration + 1
        end_iteration = start_iteration + self.cfg.max_iterations
        collect_time = world_model_time = policy_time = 0.0
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
            num_collected_steps = 0
            self._synchronize_cuda()
            for iteration in range(start_iteration, end_iteration):
                start = time.perf_counter()
                self.step(logger)
                self._synchronize_cuda()
                num_collected_steps += self.adapter.num_envs
                collect_time += time.perf_counter() - start
                model_update_time, policy_update_time = self._train_updates(iteration, logger)
                world_model_time += model_update_time
                policy_time += policy_update_time

                self._iteration = iteration
                final_iteration = iteration + 1 == end_iteration
                if iteration % self.cfg.log_interval == 0 or final_iteration:
                    logger.log(
                        iteration=iteration,
                        total_iterations=end_iteration,
                        collect_time=collect_time,
                        world_model_time=world_model_time,
                        policy_time=policy_time,
                        num_collected_steps=num_collected_steps,
                    )
                    collect_time = world_model_time = policy_time = 0.0
                    num_collected_steps = 0
                if iteration % self.cfg.save_interval == 0 or final_iteration:
                    self._logger_state = logger.state_dict()
                    self.save(log_dir / f"model_{iteration}.pt", infos)
                    _save_checkpoint(log_dir / "world_model.pt", self._world_model_checkpoint())
        return log_dir

    def _world_model_checkpoint(self) -> dict:
        return {
            "format_version": 1,
            "iter": self._iteration,
            "config": {"dynamics_hidden_dims": self.cfg.dynamics_hidden_dims},
            "dynamics": self.dynamics.state_dict(),
            "representation": self.cfg.representation,
            "joint_names": list(self.adapter.action_term._joint_names),
            "body_names": list(self.adapter.motion_command.cfg.body_names),
        }

    def save(self, path: Path, infos: dict | None = None) -> None:
        """Write actor, world model, optimizers and configuration to a checkpoint."""
        checkpoint = {
            **self._world_model_checkpoint(),
            "infos": infos,
            "logger": self._logger_state,
            "config": {
                **asdict(self.cfg),
                "num_goal_frames": self.num_goal_frames,
                "goal_frame_stride": self.goal_frame_stride,
            },
            "actor": self.actor.state_dict(),
            "actor_optimizers": [optimizer.state_dict() for optimizer in self.policy_learner.actor_optimizers],
            "world_model_optimizers": [optimizer.state_dict() for optimizer in self.world_model_learner.optimizers],
            "actor_obs_mode": "proprio",
            "task_names": list(self.actor.task_names),
            "slot_mix": dict(self.cfg.slot_mix),
        }
        if isinstance(self.policy_learner, ActorCriticPolicyLearner):
            checkpoint["critic"] = self.policy_learner.critic.state_dict()
            checkpoint["critic_optimizers"] = [
                optimizer.state_dict() for optimizer in self.policy_learner.critic_optimizers
            ]
        _save_checkpoint(path, checkpoint)

    def _validate_checkpoint_layout(self, checkpoint: dict) -> None:
        if checkpoint["representation"] != self.cfg.representation:
            raise ValueError("Checkpoint representation must match the training agent")
        if checkpoint["joint_names"] != list(self.adapter.action_term._joint_names):
            raise ValueError("Checkpoint joints must match the training task's action order")
        if checkpoint["body_names"] != list(self.adapter.motion_command.cfg.body_names):
            raise ValueError("Checkpoint links must match the training task's tracking order")

    def load_world_model(self, checkpoint: dict) -> None:
        """Initialize dynamics and its normalizers from a standalone or full checkpoint."""
        self._validate_checkpoint_layout(checkpoint)
        if tuple(checkpoint["config"]["dynamics_hidden_dims"]) != self.cfg.dynamics_hidden_dims:
            raise ValueError("Checkpoint dynamics hidden dimensions must match the training agent")
        state = checkpoint["dynamics"]
        target_state = self.dynamics.state_dict()
        for name in ("action_scale", "action_offset", "representation.default_joint_pos"):
            if not torch.equal(state[name].to(target_state[name]), target_state[name]):
                raise ValueError(f"Checkpoint dynamics {name} must match the training task")
        self.dynamics.load_state_dict(state)

    def load(self, checkpoint: dict) -> None:
        """Restore weights and optimizer states before collecting a new replay buffer."""
        self._validate_checkpoint_layout(checkpoint)
        saved_cfg = checkpoint["config"]
        if (saved_cfg["num_goal_frames"], saved_cfg["goal_frame_stride"]) != (
            self.num_goal_frames,
            self.goal_frame_stride,
        ) or saved_cfg["history_length"] != self.cfg.history_length:
            raise ValueError("Checkpoint observation history and Motion window must match the training task")
        if checkpoint["actor_obs_mode"] != "proprio" or checkpoint["task_names"] != list(self.actor.task_names):
            raise ValueError("Checkpoint actor observations and encoders must match the training agent")
        has_critic = isinstance(self.policy_learner, ActorCriticPolicyLearner)
        if has_critic != ("critic" in checkpoint):
            raise ValueError("Checkpoint policy learner must match the training agent")

        self.actor.load_state_dict(checkpoint["actor"])
        self.dynamics.load_state_dict(checkpoint["dynamics"])
        for optimizer, state in zip(self.policy_learner.actor_optimizers, checkpoint["actor_optimizers"], strict=True):
            optimizer.load_state_dict(state)
        for optimizer, state in zip(
            self.world_model_learner.optimizers, checkpoint["world_model_optimizers"], strict=True
        ):
            optimizer.load_state_dict(state)
        if isinstance(self.policy_learner, ActorCriticPolicyLearner):
            self.policy_learner.critic.load_state_dict(checkpoint["critic"])
            for optimizer, state in zip(
                self.policy_learner.critic_optimizers, checkpoint["critic_optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
        self._logger_state = checkpoint.get("logger", {})
        self._iteration = checkpoint["iter"] if "iter" in checkpoint else checkpoint["iteration"]

    def train(self, log_dir: Path, infos: dict | None = None) -> Path:
        """Run the training loop in the supplied run directory."""
        return self.run(log_dir, infos)
