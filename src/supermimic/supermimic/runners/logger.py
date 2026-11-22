"""Report SuperMimic training and Isaac Lab episode statistics."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta
from statistics import mean
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """Write aligned console summaries and TensorBoard scalars for one run."""

    def __init__(
        self,
        writer: SummaryWriter,
        run_name: str,
        task: str | None,
        num_envs: int,
        device: torch.device,
        start_iteration: int,
        end_iteration: int,
    ) -> None:
        self._writer = writer
        self._run_name = run_name
        self._task = task
        self._num_envs = num_envs
        self._start_iteration = start_iteration
        self._end_iteration = end_iteration
        self._last_log_iteration = start_iteration - 1
        self._elapsed = 0.0
        self._episode_returns = deque[float](maxlen=100)
        self._episode_lengths = deque[float](maxlen=100)
        self._running_returns = torch.zeros(num_envs, device=device)
        self._running_lengths = torch.zeros_like(self._running_returns)
        self._episode_metrics: dict[str, list[torch.Tensor]] = defaultdict(list)

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Record completed episodes and sample reset metrics only on fresh resets."""
        self._running_returns += rewards.reshape(-1)
        self._running_lengths += 1
        done_ids = dones.reshape(-1).nonzero(as_tuple=True)[0]
        if done_ids.numel() == 0:
            return

        returns, lengths = torch.stack((self._running_returns[done_ids], self._running_lengths[done_ids])).tolist()
        self._episode_returns.extend(returns)
        self._episode_lengths.extend(lengths)
        self._running_returns[done_ids] = 0
        self._running_lengths[done_ids] = 0

        for name, value in extras.get("episode", extras.get("log", {})).items():
            tag = name if "/" in name else f"Episode/{name}"
            metric = torch.as_tensor(value, dtype=torch.float32, device=rewards.device).detach().reshape(-1)
            if metric.numel() > 0:
                self._episode_metrics[tag].append(metric.clone())

    def log(
        self,
        iteration: int,
        collect_time: float,
        world_model_time: float,
        policy_time: float,
        losses: dict[str, torch.Tensor],
        env_reward: torch.Tensor,
        replay_size: int,
        replay_capacity: int,
        action_noise: float,
    ) -> None:
        """Report timings for the logging window and the latest learner metrics."""
        window_iterations = iteration - self._last_log_iteration
        self._last_log_iteration = iteration
        iterations = iteration - self._start_iteration + 1
        window_time = collect_time + world_model_time + policy_time
        self._elapsed += window_time
        total_steps = iterations * self._num_envs
        fps = int(window_iterations * self._num_envs / max(window_time, 1e-9))
        iteration_time = window_time / window_iterations
        eta = self._elapsed / iterations * (self._end_iteration - iteration - 1)

        task_metrics = {tag: torch.cat(values).mean() for tag, values in self._episode_metrics.items()}
        tensors = {"Train/env_reward": env_reward, **losses, **task_metrics}
        scalars = dict(zip(tensors, torch.stack(list(tensors.values())).tolist(), strict=True))
        perf = {
            "Perf/total_steps": total_steps,
            "Perf/total_fps": fps,
            "Perf/collection_time": collect_time / window_iterations,
            "Perf/wm_learning_time": world_model_time / window_iterations,
            "Perf/policy_learning_time": policy_time / window_iterations,
        }
        episode = {}
        if self._episode_returns:
            episode["Train/mean_reward"] = mean(self._episode_returns)
            episode["Train/mean_episode_length"] = mean(self._episode_lengths)
        rollout = {"Replay/size": replay_size, "Policy/action_noise_std": action_noise}
        logged_scalars = perf | scalars | episode | rollout
        for tag, value in logged_scalars.items():
            self._writer.add_scalar(tag, value, iteration)

        rows = [("Run name", self._run_name)]
        if self._task:
            rows.append(("Task", self._task))
        rows += [
            ("Total steps", str(total_steps)),
            ("Steps per second", str(fps)),
            ("Collection time", f"{perf['Perf/collection_time']:.3f}s"),
            ("WM Learning time", f"{perf['Perf/wm_learning_time']:.3f}s"),
            ("Policy Learning time", f"{perf['Perf/policy_learning_time']:.3f}s"),
            ("Replay buffer", f"{replay_size}/{replay_capacity}"),
            ("Mean step reward", f"{scalars['Train/env_reward']:.5g}"),
            ("Action noise std", f"{action_noise:.5g}"),
        ]
        rows.extend((tag, f"{scalars[tag]:.5g}") for tag in losses)
        if episode:
            rows += [
                ("Mean reward", f"{episode['Train/mean_reward']:.2f}"),
                ("Mean episode length", f"{episode['Train/mean_episode_length']:.2f}"),
            ]
        rows.extend((tag, f"{scalars[tag]:.5g}") for tag in task_metrics)

        title = f" Learning iteration {iteration}/{self._end_iteration} "
        lines = ["#" * 80, title.center(80), ""]
        lines.extend(f"{name + ':':>40} {value}" for name, value in rows)
        lines += [
            "-" * 80,
            f"{'Iteration time:':>40} {iteration_time:.2f}s",
            f"{'Time elapsed:':>40} {timedelta(seconds=int(self._elapsed))}",
            f"{'ETA:':>40} {timedelta(seconds=int(eta))}",
        ]
        print("\n".join(lines), flush=True)
        self._episode_metrics.clear()
