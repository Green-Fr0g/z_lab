"""Export the SuperAMP velocity actor for deployment."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import onnx
import torch
from torch import nn

from superamp.envs import IsaacLabSuperAmpAdapter
from superamp.models import VelocityActor


class _VelocityPolicy(nn.Module):
    """Apply the trained velocity policy to term-major deployment observations."""

    def __init__(self, actor: VelocityActor, history_length: int, term_widths: tuple[int, ...]) -> None:
        super().__init__()
        self.obs_normalizer = actor.obs_normalizer
        self.goal_normalizer = actor.goal_normalizer
        self.network = actor.network
        self.history_length = history_length
        self.history_size = sum(term_widths) * history_length
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
        return self.network(torch.cat((normalized_history.flatten(start_dim=1), self.goal_normalizer(goal)), dim=-1))


@torch.no_grad()
def _write_policy(
    module: _VelocityPolicy,
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
    actor: VelocityActor, adapter: IsaacLabSuperAmpAdapter, path: Path, run_name: str, history_length: int
) -> list[Path]:
    """Write the velocity policy and its deployment metadata in both formats."""
    manager = adapter.env.observation_manager
    term_widths = tuple(width for (width,) in manager.group_obs_term_dim["actor"])
    policy = _VelocityPolicy(actor, history_length, term_widths)
    sample = torch.zeros(1, policy.history_size + adapter.observation_dims["goal"])
    scale, offset = adapter.action_scale_offset()
    ranges = adapter.env.cfg.commands.base_velocity.ranges
    metadata = {
        "run_name": run_name,
        "task": "velocity",
        "joint_names": list(adapter.action_term._joint_names),
        "body_names": list(adapter.env.cfg.amp_body_names),
        "anchor_body_name": adapter.env.cfg.anchor_link_name,
        "action_scale": scale.tolist(),
        "action_offset": offset.tolist(),
        "quaternion_order": "xyzw",
        "actor_obs_mode": "proprio",
        "history_length": history_length,
        "observation_names": list(manager.active_terms["actor"]) + ["command"],
        "observation_history_lengths": [history_length] * len(term_widths) + [1],
        "observation_term_widths": [*term_widths, adapter.observation_dims["goal"]],
        "observation_layout": "term-major proprioception (oldest to newest) then body-frame velocity command",
        "command_names": ["lin_vel_x", "lin_vel_y", "ang_vel_z"],
        "velocity_command_ranges": [ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z],
    }
    path.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path) as temporary_dir:
        staging = Path(temporary_dir)
        onnx_path = staging / "policy_velocity.onnx"
        jit_path = staging / "policy_velocity.pt"
        _write_policy(policy, sample, onnx_path, jit_path, metadata)
        return [file.replace(path / file.name) for file in (onnx_path, jit_path)]
