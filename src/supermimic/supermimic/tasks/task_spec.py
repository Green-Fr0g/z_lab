"""Task goal construction shared by training and deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict

from supermimic.utils.math import quat_apply_inverse


@dataclass(frozen=True)
class VelocityTaskCfg:
    """Body-frame planar velocity command ranges and training perturbations."""

    lin_vel_x_range: tuple[float, float] = (-1.5, 1.5)
    lin_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z_range: tuple[float, float] = (-3.14, 3.14)
    window_frames: int = 25
    command_noise: tuple[float, float, float] = (0.5, 0.5, 0.785)


def build_actor_obs_td(
    goals: dict[str, torch.Tensor], history: torch.Tensor, critic: torch.Tensor | None = None
) -> TensorDict:
    """Pack full-batch task goals with actor history and privileged critic inputs."""
    observations = {"history": history}
    if critic is not None:
        observations["critic"] = critic
    observations.update({f"goal_{name}": goal for name, goal in goals.items()})
    return TensorDict(observations, batch_size=[history.shape[0]], device=history.device)


def advance_proprio_history(history: torch.Tensor, observation: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    """Append the newest observation, repeating it on environment resets."""
    latest = observation.unsqueeze(1)
    if history.shape[1] == 1:
        return latest
    shifted = torch.cat((history[:, 1:], latest), dim=1)
    return torch.where(dones.view(-1, 1, 1), latest, shifted)


def velocity_goal_from_motion(
    motion: Any,
    motion_times: torch.Tensor,
    root_quat_xyzw: torch.Tensor,
    cfg: VelocityTaskCfg,
    add_noise: bool = True,
) -> torch.Tensor:
    """Project centered-mean reference velocity into the current root frame."""
    if cfg.window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if root_quat_xyzw.shape[:-1] != motion_times.shape or root_quat_xyzw.shape[-1] != 4:
        raise ValueError("root quaternions must have shape [batch, 4]")

    if cfg.window_frames == 1:
        linear_world = motion.root_lin_vel_w[motion_times]
        angular_world = motion.root_ang_vel_w[motion_times]
    else:
        half_window = cfg.window_frames // 2
        offsets = torch.arange(cfg.window_frames, device=motion_times.device) - half_window
        indices = (motion_times[:, None] + offsets).clamp(0, motion.time_step_total - 1)
        linear_world = motion.root_lin_vel_w[indices].mean(dim=1)
        angular_world = motion.root_ang_vel_w[indices].mean(dim=1)

    linear_body = quat_apply_inverse(root_quat_xyzw, linear_world)
    angular_body = quat_apply_inverse(root_quat_xyzw, angular_world)
    command = torch.stack((linear_body[:, 0], linear_body[:, 1], angular_body[:, 2]), dim=-1)
    lower = command.new_tensor((cfg.lin_vel_x_range[0], cfg.lin_vel_y_range[0], cfg.ang_vel_z_range[0]))
    upper = command.new_tensor((cfg.lin_vel_x_range[1], cfg.lin_vel_y_range[1], cfg.ang_vel_z_range[1]))
    if add_noise:
        noise_half_width = command.new_tensor(cfg.command_noise)
        command = command + (torch.rand_like(command) * 2.0 - 1.0) * noise_half_width
    return torch.maximum(torch.minimum(command, upper), lower)
