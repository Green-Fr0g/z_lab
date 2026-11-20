"""Export task-specific SuperMimic actors for deployment."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import nn

from supermimic.algorithms.models import MultiEncoderAE
from supermimic.tasks import VelocityTaskCfg


class _SingleTaskActor(nn.Module):
    """Flattened observation wrapper for one encoder and the shared decoder."""

    def __init__(self, actor: MultiEncoderAE, task_name: str, term_widths: tuple[int, ...]) -> None:
        super().__init__()
        self.obs_normalizer = actor.obs_normalizer
        self.goal_normalizer = actor.goal_normalizers[task_name]
        self.encoder = actor.encoders[task_name]
        self.decoder = actor.decoder
        self.history_length = actor.history_length
        self.history_size = actor.obs_dim * actor.history_length
        self.term_widths = term_widths
        self.term_history_sizes = tuple(width * self.history_length for width in term_widths)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        terms = observation[:, : self.history_size].split(self.term_history_sizes, dim=-1)
        history = torch.cat(
            [term.reshape(-1, self.history_length, width) for term, width in zip(terms, self.term_widths, strict=True)],
            dim=-1,
        )
        goal = observation[:, self.history_size :]
        normalized_history = self.obs_normalizer(history)
        latent = torch.tanh(self.encoder(self.goal_normalizer(goal)))
        return self.decoder(torch.cat((latent, normalized_history.flatten(start_dim=1)), dim=-1))


class _MotionDeployActor(nn.Module):
    """Motion actor plus frame-indexed reference data for tracking deployment."""

    def __init__(self, actor: MultiEncoderAE, motion_command: Any, term_widths: tuple[int, ...]) -> None:
        super().__init__()
        self.actor = _SingleTaskActor(actor, "motion", term_widths)
        motion = motion_command.motion
        joint_indices = motion_command.joint_indexes
        frames = torch.arange(motion.time_step_total, device=motion.joint_pos.device)
        body_pos, body_quat, body_lin_vel, body_ang_vel = motion.body_state(frames)
        self.register_buffer("joint_pos", motion.joint_pos[:, joint_indices].detach().cpu())
        self.register_buffer("joint_vel", motion.joint_vel[:, joint_indices].detach().cpu())
        self.register_buffer("body_pos_w", body_pos.detach().cpu())
        self.register_buffer("body_quat_w", body_quat.detach().cpu())
        self.register_buffer("body_lin_vel_w", body_lin_vel.detach().cpu())
        self.register_buffer("body_ang_vel_w", body_ang_vel.detach().cpu())
        self.num_frames = int(self.joint_pos.shape[0])

    def forward(self, observation: torch.Tensor, time_step: torch.Tensor) -> tuple[torch.Tensor, ...]:
        frame = time_step.long().reshape(-1).clamp(0, self.num_frames - 1)
        return (
            self.actor(observation),
            self.joint_pos[frame],
            self.joint_vel[frame],
            self.body_pos_w[frame],
            self.body_quat_w[frame],
            self.body_lin_vel_w[frame],
            self.body_ang_vel_w[frame],
        )


def _write_onnx(
    module: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    path: Path,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_module = copy.deepcopy(module).cpu().eval()
    torch.onnx.export(
        cpu_module,
        sample_inputs,
        str(path),
        export_params=True,
        opset_version=18,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={},
        dynamo=False,
    )
    model = onnx.load(str(path))
    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = _metadata_value(value)
        model.metadata_props.append(entry)
    onnx.save(model, str(path))


def _metadata_value(value: Any) -> str:
    """Encode metadata lists with the upstream comma/semicolon convention."""
    if isinstance(value, (list, tuple)):
        entries = []
        for item in value:
            if isinstance(item, (list, tuple)):
                entries.append(";".join(_metadata_scalar(scalar) for scalar in item))
            else:
                entries.append(_metadata_scalar(item))
        return ",".join(entries)
    return str(value)


def _metadata_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def export_actor_as_onnx(
    actor: MultiEncoderAE,
    env: Any,
    path: str | Path,
    run_name: str,
    velocity_cfg: VelocityTaskCfg,
) -> list[Path]:
    """Write motion-tracking and velocity-command ONNX policies."""
    if set(actor.task_names) != {"motion", "velocity"}:
        raise ValueError("deployment export requires motion and velocity encoders")
    env = env.unwrapped
    output_dir = Path(path)
    motion_command = env.command_manager.get_term("motion")
    action_term = env.action_manager.get_term("joint_pos")
    observations = env.observation_manager
    actor_terms = list(observations.active_terms["actor"])
    term_widths = tuple(width for (width,) in observations.group_obs_term_dim["actor"])
    if sum(term_widths) != actor.obs_dim:
        raise ValueError("Actor input width does not match task proprioception")
    motion_terms = [
        (name, width)
        for name, (width,) in zip(
            observations.active_terms["goal"], observations.group_obs_term_dim["goal"], strict=True
        )
        if name != "motion_anchor_pos_b"
    ]
    goal_dims = {task.name: task.goal_dim for task in actor.task_specs}
    if sum(width for _, width in motion_terms) != goal_dims["motion"]:
        raise ValueError("Motion encoder input width does not match task goals")
    actor_history_lengths = [actor.history_length] * len(actor_terms)
    action_scale = action_term._scale
    action_offset = action_term._offset
    if isinstance(action_scale, torch.Tensor):
        action_scale = action_scale[0] if action_scale.ndim > 1 else action_scale
        action_scale = action_scale.detach().cpu().tolist()
    if isinstance(action_offset, torch.Tensor):
        action_offset = action_offset[0] if action_offset.ndim > 1 else action_offset
        action_offset = action_offset.detach().cpu().tolist()
    else:
        action_offset = [float(action_offset)] * actor.decoder[-1].out_features
    if not isinstance(action_scale, list):
        action_scale = [float(action_scale)] * actor.decoder[-1].out_features

    common_metadata = {
        "run_name": run_name,
        "joint_names": list(action_term._joint_names),
        "body_names": list(motion_command.cfg.body_names),
        "anchor_body_name": motion_command.cfg.anchor_body_name,
        "action_scale": action_scale,
        "action_offset": action_offset,
        "quaternion_order": "xyzw",
        "actor_obs_mode": "proprio",
        "history_length": actor.history_length,
        "goal_stride": env.cfg.observations.goal.command.params["frame_stride"],
    }
    motion_policy = _MotionDeployActor(actor, motion_command, term_widths)
    motion_obs_dim = actor.obs_dim * actor.history_length + goal_dims["motion"]
    motion_path = output_dir / "policy_motion.onnx"
    _write_onnx(
        motion_policy,
        (torch.zeros(1, motion_obs_dim), torch.zeros(1, 1)),
        ["obs", "time_step"],
        [
            "actions",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ],
        motion_path,
        {
            **common_metadata,
            "task": "motion",
            "observation_names": actor_terms + [name for name, _ in motion_terms],
            "observation_history_lengths": actor_history_lengths + [1] * len(motion_terms),
            "observation_term_widths": list(term_widths) + [width for _, width in motion_terms],
            "observation_layout": "term-major proprioception (oldest to newest) followed by motion goal",
        },
    )

    velocity_actor = _SingleTaskActor(actor, "velocity", term_widths)
    velocity_obs_dim = actor.obs_dim * actor.history_length + goal_dims["velocity"]
    velocity_path = output_dir / "policy_velocity.onnx"
    _write_onnx(
        velocity_actor,
        (torch.zeros(1, velocity_obs_dim),),
        ["obs"],
        ["actions"],
        velocity_path,
        {
            **common_metadata,
            "task": "velocity",
            "observation_names": actor_terms + ["command"],
            "observation_history_lengths": actor_history_lengths + [1],
            "observation_term_widths": list(term_widths) + [goal_dims["velocity"]],
            "observation_layout": "term-major proprioception (oldest to newest) then body-frame velocity command",
            "command_names": ["lin_vel_x", "lin_vel_y", "ang_vel_z"],
            "velocity_command_ranges": [
                velocity_cfg.lin_vel_x_range,
                velocity_cfg.lin_vel_y_range,
                velocity_cfg.ang_vel_z_range,
            ],
        },
    )
    return [motion_path, velocity_path]
