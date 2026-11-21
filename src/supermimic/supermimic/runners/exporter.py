"""Export task-specific SuperMimic actors for deployment."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
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


@torch.no_grad()
def _write_policy(
    module: _SingleTaskActor,
    sample_observation: torch.Tensor,
    onnx_path: Path,
    jit_path: Path,
    metadata: dict[str, Any],
) -> None:
    cpu_module = copy.deepcopy(module).cpu().eval()
    torch.onnx.export(
        cpu_module,
        (sample_observation,),
        str(onnx_path),
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
        dynamo=False,
    )
    model = onnx.load(str(onnx_path))
    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = _metadata_value(value)
        model.metadata_props.append(entry)
    onnx.save(model, str(onnx_path))

    traced = torch.jit.trace(cpu_module, (sample_observation,))
    torch.jit.save(traced, str(jit_path), _extra_files={"metadata.json": json.dumps(metadata)})


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


def export_actor(
    actor: MultiEncoderAE,
    env: Any,
    path: str | Path,
    run_name: str,
    velocity_cfg: VelocityTaskCfg,
) -> list[Path]:
    """Write one ONNX and one TorchScript policy per encoder."""
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

    goal_params = env.cfg.observations.goal.command.params
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
        "goal_stride": goal_params["frame_stride"],
    }
    task_metadata = {
        "motion": {
            "num_goal_frames": goal_params["num_frames"],
            "observation_names": actor_terms + [name for name, _ in motion_terms],
            "observation_history_lengths": actor_history_lengths + [1] * len(motion_terms),
            "observation_term_widths": list(term_widths) + [width for _, width in motion_terms],
            "observation_layout": "term-major proprioception (oldest to newest) followed by motion goal",
        },
        "velocity": {
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
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    with TemporaryDirectory(dir=output_dir) as temporary_dir:
        staging_dir = Path(temporary_dir)
        for task_name, metadata in task_metadata.items():
            policy = _SingleTaskActor(actor, task_name, term_widths)
            sample_observation = torch.zeros(1, policy.history_size + goal_dims[task_name])
            onnx_path = staging_dir / f"policy_{task_name}.onnx"
            jit_path = staging_dir / f"policy_{task_name}.pt"
            _write_policy(
                policy,
                sample_observation,
                onnx_path,
                jit_path,
                {**common_metadata, "task": task_name, **metadata},
            )
            exported.extend((onnx_path, jit_path))
        return [path.replace(output_dir / path.name) for path in exported]
