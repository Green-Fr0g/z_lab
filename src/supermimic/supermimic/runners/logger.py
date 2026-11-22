"""Report SuperMimic training and Isaac Lab episode statistics."""

from __future__ import annotations

import time
from collections import defaultdict, deque
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
        max_episode_length: int,
        device: torch.device,
    ) -> None:
        self._writer = writer
        self._run_name = run_name
        self._task = task
        self._num_envs = num_envs
        self._max_episode_length = max_episode_length
        self._total_steps = 0
        self._total_time = 0.0
        self._episode_returns = deque[float](maxlen=100)
        self._episode_lengths = deque[float](maxlen=100)
        self._running_returns = torch.zeros(num_envs, device=device)
        self._running_lengths = torch.zeros_like(self._running_returns)
        self._episode_metrics: dict[str, list[torch.Tensor]] = defaultdict(list)

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Record completed episodes and sample reset metrics only on fresh resets."""
        self._running_returns += rewards.reshape(-1)
        self._running_lengths += 1
        flush_mask = dones.reshape(-1) | (self._running_lengths >= self._max_episode_length)
        done_ids = flush_mask.nonzero(as_tuple=True)[0]
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
        total_iterations: int,
        collect_time: float,
        world_model_time: float,
        policy_time: float,
        losses: dict[str, torch.Tensor],
        num_collected_steps: int,
    ) -> None:
        """Report metrics using the same collection window units as the upstream logger."""
        iteration_time = collect_time + world_model_time + policy_time
        self._total_steps += num_collected_steps
        self._total_time += iteration_time
        fps = int(num_collected_steps / (iteration_time + 1e-6))

        task_metrics = {tag: torch.cat(values).mean() for tag, values in self._episode_metrics.items()}
        perf = {
            "Perf/total_fps": fps,
            "Perf/collection_time": collect_time,
            "Perf/wm_learning_time": world_model_time,
            "Perf/policy_learning_time": policy_time,
        }
        episode = {}
        if self._episode_returns:
            episode["Train/mean_reward"] = mean(self._episode_returns)
            episode["Train/mean_episode_length"] = mean(self._episode_lengths)
        scalar_losses = {
            name: value.item() if isinstance(value, torch.Tensor) else value for name, value in losses.items()
        }
        logged_scalars = scalar_losses | perf | episode | task_metrics
        for tag, value in logged_scalars.items():
            self._writer.add_scalar(tag, value, iteration)

        rows = [("Run name", self._run_name)]
        rows += [
            ("Total steps", str(self._total_steps)),
            ("Steps per second", str(fps)),
            ("Collection time", f"{collect_time:.3f}s"),
            ("WM Learning time", f"{world_model_time:.3f}s"),
            ("Policy Learning time", f"{policy_time:.3f}s"),
        ]
        rows.extend((tag, f"{value:.4f}") for tag, value in scalar_losses.items())
        if episode:
            rows += [
                ("Mean reward", f"{episode['Train/mean_reward']:.2f}"),
                ("Mean episode length", f"{episode['Train/mean_episode_length']:.2f}"),
            ]
        rows.extend((tag, f"{value.item():.4f}") for tag, value in task_metrics.items())

        title = f" Learning iteration {iteration}/{total_iterations} "
        lines = ["#" * 80, title.center(80), ""]
        lines.extend(f"{name + ':':>40} {value}" for name, value in rows)
        eta_seconds = self._total_time / max(iteration + 1, 1) * max(total_iterations - iteration - 1, 0)
        lines += [
            "-" * 80,
            f"{'Iteration time:':>40} {iteration_time:.2f}s",
            f"{'Time elapsed:':>40} {time.strftime('%H:%M:%S', time.gmtime(self._total_time))}",
            f"{'ETA:':>40} {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}",
        ]
        print("\n".join(lines), flush=True)
        self._episode_metrics.clear()
