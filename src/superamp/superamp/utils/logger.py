"""Report SuperAMP training and Isaac Lab episode statistics."""

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
        state: dict,
    ) -> None:
        self._writer = writer
        self._run_name = run_name
        self._task = task
        self._start_iteration = start_iteration
        self._total_steps = state.get("total_steps", 0)
        self._steps_known = state.get("steps_known", start_iteration == 0)
        self._total_time = 0.0
        self._episode_returns = deque[float](state.get("episode_returns", ()), maxlen=100)
        self._episode_lengths = deque[float](state.get("episode_lengths", ()), maxlen=100)
        self._running_returns = torch.zeros(num_envs, device=device)
        self._running_lengths = torch.zeros_like(self._running_returns)
        self._update_metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._last_episode_metrics: dict[str, float] = dict(state.get("last_episode_metrics", {}))
        self._episode_metrics: dict[str, list[torch.Tensor]] = defaultdict(list)

    def state_dict(self) -> dict:
        """Save cumulative collection counts and completed episodes across simulator restarts."""
        return {
            "total_steps": self._total_steps,
            "steps_known": self._steps_known,
            "episode_returns": list(self._episode_returns),
            "episode_lengths": list(self._episode_lengths),
            "last_episode_metrics": self._last_episode_metrics | self._window_episode_means(),
        }

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Record completed episodes and sample reset metrics only on fresh resets."""
        self._running_returns += rewards.reshape(-1)
        self._running_lengths += 1
        self._total_steps += rewards.numel()
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

    def _window_episode_means(self) -> dict[str, float]:
        return {tag: torch.cat(values).mean().item() for tag, values in self._episode_metrics.items()}

    def process_update(self, metrics: dict[str, torch.Tensor]) -> None:
        """Collect each reported scalar once per learner update."""
        for name, value in metrics.items():
            self._update_metrics[name].append(torch.as_tensor(value).detach().clone())

    def log(
        self,
        iteration: int,
        total_iterations: int,
        collect_time: float,
        world_model_time: float,
        policy_time: float,
        num_collected_steps: int,
    ) -> None:
        """Report metrics using the same collection window units as the upstream logger."""
        iteration_time = collect_time + world_model_time + policy_time
        self._total_time += iteration_time
        fps = int(num_collected_steps / (iteration_time + 1e-6))

        task_metrics = self._window_episode_means()
        self._last_episode_metrics.update(task_metrics)
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
            name: torch.stack(values).float().mean().item() for name, values in self._update_metrics.items()
        }
        logged_scalars = scalar_losses | perf | episode | task_metrics
        for tag, value in logged_scalars.items():
            self._writer.add_scalar(tag, value, iteration)

        rows = [("Run name", self._run_name)]
        if self._task:
            rows.append(("Task", self._task))
        rows += [
            ("Total steps" if self._steps_known else "Steps since legacy checkpoint", str(self._total_steps)),
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
        for tag, value in self._last_episode_metrics.items():
            suffix = "" if tag in task_metrics else " (last observed)"
            rows.append((tag, f"{value:.4f}{suffix}"))

        if not scalar_losses:
            rows.append(("Learner metrics", "No update in this window"))
        if not self._last_episode_metrics:
            rows.append(("Episode metrics", "Waiting for first reset metrics"))

        title = f" Learning iteration {iteration}/{total_iterations - 1} "
        lines = ["#" * 80, title.center(80), ""]
        lines.extend(f"{name + ':':>40} {value}" for name, value in rows)
        completed = iteration - self._start_iteration + 1
        eta_seconds = self._total_time / completed * (total_iterations - iteration - 1)
        lines += [
            "-" * 80,
            f"{'Log window time:':>40} {iteration_time:.2f}s",
            f"{'Session training time:':>40} {timedelta(seconds=int(self._total_time))}",
            f"{'ETA:':>40} {timedelta(seconds=int(eta_seconds))}",
        ]
        print("\n".join(lines), flush=True)
        self._update_metrics.clear()
        self._episode_metrics.clear()
