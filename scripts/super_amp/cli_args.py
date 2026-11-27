"""Resolve registered SuperAMP task configurations and checkpoint paths."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args

from isaaclab_rl.entrypoints.common import normalize_task_name, resolve_checkpoint_selector

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import z_lab.tasks  # noqa: F401


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add task, agent, run selection, and Isaac Lab launcher flags."""
    parser.add_argument("--task", required=True)
    parser.add_argument("--agent", default="superamp_cfg_entry_point")
    parser.add_argument("--checkpoint", help="Checkpoint file or 'latest' in the experiment directory.")
    parser.add_argument("--experiment_name", "--experiment-name")
    parser.add_argument("--run-dir", type=Path, default=Path("logs/superamp"))
    parser.add_argument("--num_envs", "--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--motion-file", type=Path)
    parser.add_argument("--physics", help="Isaac Lab physics preset for the task.")
    add_launcher_args(parser)


def parse_task_configs(parser: argparse.ArgumentParser, *, play: bool = False):
    """Resolve presets, nested overrides, and explicit CLI flags."""
    args, overrides = setup_preset_cli(parser)
    if args.physics is not None:
        overrides = [override for override in overrides if not override.startswith("physics=")]
        overrides.append(f"physics={args.physics}")
        args.physics = None
    args.task = normalize_task_name(args.task)
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=play, overrides=overrides)
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.seed is not None:
        agent_cfg.seed = args.seed
    env_cfg.seed = agent_cfg.seed
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if getattr(args, "device_explicit", False):
        env_cfg.sim.device = args.device
    else:
        args.device = env_cfg.sim.device
    if args.motion_file is not None:
        env_cfg.events.reset_from_expert.params["motion_file"] = str(args.motion_file.resolve())
    if env_cfg.scene.num_envs <= 0:
        parser.error("--num_envs must be positive")
    return args, env_cfg, agent_cfg


def checkpoint_path(args: argparse.Namespace, log_root: Path, selection: str, *, world_model: bool = False) -> Path:
    """Find an explicit checkpoint or resolve the newest run's checkpoint."""
    if selection == "latest":
        selection = resolve_checkpoint_selector(
            str(log_root),
            "best" if world_model else "latest",
            library="superamp",
            task=args.task,
            checkpoint_pattern=r"(?:model_\d+|world_model)\.pt" if world_model else r"model_\d+\.pt",
            preferred_checkpoint_pattern=r"world_model\.pt" if world_model else None,
            metadata={"agent": args.agent},
        )
    path = Path(selection).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SuperAMP checkpoint not found: {path}")
    return path
