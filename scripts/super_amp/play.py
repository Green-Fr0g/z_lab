"""Play a trained SuperAMP velocity policy in Isaac Lab."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch
from cli_args import add_common_args, checkpoint_path, parse_task_configs
from superamp.envs import IsaacLabSuperAmpAdapter
from superamp.runners import SuperAmpRunnerCfg
from superamp.runners.exporter import export_actor
from superamp.runners.runner import advance_proprio_history, build_actor, build_actor_observations

from isaaclab.app import launch_simulation


@torch.no_grad()
def run_playback(
    adapter: IsaacLabSuperAmpAdapter, checkpoint: dict, num_steps: int | None, checkpoint_file: Path | None = None
) -> None:
    """Run the actor with the checkpoint's observation history and network layout."""
    if checkpoint["joint_names"] != list(adapter.action_term._joint_names):
        raise ValueError("Checkpoint joint order does not match the playback task")
    cfg = SuperAmpRunnerCfg(
        representation=checkpoint["representation"],
        history_length=checkpoint["config"]["history_length"],
        actor_hidden_dims=tuple(checkpoint["config"]["actor_hidden_dims"]),
    )
    actor = build_actor(cfg, adapter)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    if checkpoint_file is not None:
        files = export_actor(
            actor, adapter, checkpoint_file.parent / "exported", checkpoint_file.stem, cfg.history_length
        )
        print("[SuperAMP] deployment files written to " + ", ".join(map(str, files)))
    env = adapter.env
    observations, _ = env.reset()
    history = observations["actor"].unsqueeze(1).repeat(1, cfg.history_length, 1)
    step = 0
    while (num_steps is None or step < num_steps) and env.sim.is_running():
        actions = actor(build_actor_observations(history, observations["goal"]))
        observations, _, ended, timed_out, _ = env.step(actions)
        history = advance_proprio_history(history, observations["actor"], ended | timed_out)
        step += 1
    print(f"[SuperAMP] playback completed {step} steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--num-steps", type=int, help="Stop after this many simulator steps.")
    args, env_cfg, agent_cfg = parse_task_configs(parser, play=True)
    if args.num_steps is not None and args.num_steps < 0:
        parser.error("--num-steps must be non-negative")
    path = checkpoint_path(args, args.run_dir / agent_cfg.experiment_name, args.checkpoint or "latest")
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["representation"] != agent_cfg.representation:
        raise ValueError("Checkpoint representation does not match the playback task")
    env_cfg.log_dir = str(path.parent)
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            run_playback(IsaacLabSuperAmpAdapter(env), checkpoint, args.num_steps, path)
        finally:
            env.close()


if __name__ == "__main__":
    main()
