"""Play a SuperMimic actor from a local checkpoint."""

from __future__ import annotations

import argparse

import gymnasium as gym
import torch
from cli_args import add_common_args, checkpoint_path, parse_task_configs
from supermimic.algorithms.models import MultiEncoderAE, TaskSpec
from supermimic.algorithms.representations import extract_actor_goal
from supermimic.envs.isaaclab_adapter import IsaacLabSuperMimicAdapter
from supermimic.runners.exporter import export_actor
from supermimic.tasks import VelocityTaskCfg, advance_proprio_history, build_actor_obs_td

from isaaclab.app import launch_simulation
from isaaclab.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--task-encoder", choices=("motion", "velocity"), default="motion")
    parser.add_argument("--velocity-command", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    args, env_cfg, agent_cfg = parse_task_configs(parser, play=True)
    log_root = args.run_dir / agent_cfg.experiment_name
    path = checkpoint_path(args, log_root, args.checkpoint or "latest")
    if args.motion_file is None:
        training_env_path = path.parent / "params" / "env.yaml"
        if training_env_path.is_file():
            training_env_cfg = load_yaml(str(training_env_path))
            env_cfg.commands.motion.motion_file = training_env_cfg["commands"]["motion"]["motion_file"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("actor_obs_mode") != "proprio":
        raise ValueError("Playback requires a checkpoint trained with proprioception. Retrain the rep_state actor.")
    if checkpoint["representation"] != agent_cfg.representation:
        raise ValueError("Checkpoint representation must match the playback agent")
    saved_config = checkpoint["config"]
    history_length = saved_config["history_length"]
    env_cfg.log_dir = str(path.parent)
    print(f"[SuperMimic] loading {path}")
    print(f"[SuperMimic] motion library: {env_cfg.commands.motion.motion_file}")

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            env.reset()
            adapter = IsaacLabSuperMimicAdapter(env)
            observations = adapter.observations()
            obs_dim = adapter.validate_proprio_layout()
            adapter.validate_full_state_layout(agent_cfg.representation, observations["full_state"].shape[-1])
            if list(adapter.action_term._joint_names) != checkpoint["joint_names"]:
                raise ValueError("Checkpoint joints must match the playback task's action order")
            if list(adapter.motion_command.cfg.body_names) != checkpoint["body_names"]:
                raise ValueError("Checkpoint links must match the playback task's tracking order")
            frames, stride = adapter.motion_goal_window()
            if (frames, stride) != (saved_config["num_goal_frames"], saved_config["goal_frame_stride"]):
                raise ValueError("Checkpoint Motion window must match the playback task")
            goal_dim = extract_actor_goal(observations["goal"], frames).shape[-1]
            task_names = checkpoint["task_names"]
            if args.task_encoder not in task_names:
                raise ValueError(
                    f"checkpoint has no {args.task_encoder!r} encoder; train with the velocity task enabled"
                )
            task_specs = [TaskSpec("motion", goal_dim)]
            if "velocity" in task_names:
                task_specs.append(TaskSpec("velocity", 3))
            slot_mix = {name: float(name == args.task_encoder) for name in task_names}
            actor = MultiEncoderAE(
                obs_dim=obs_dim,
                action_dim=adapter.num_actions,
                task_specs=task_specs,
                slot_mix=slot_mix,
                latent_dim=saved_config["latent_dim"],
                hidden_dims=tuple(saved_config["actor_hidden_dims"]),
                history_length=history_length,
            ).to(adapter.device)
            actor.load_state_dict(checkpoint["actor"])
            actor.eval()
            if set(task_names) == {"motion", "velocity"}:
                velocity_cfg = VelocityTaskCfg(**saved_config["velocity"])
                exported = export_actor(actor, env, path.parent / "exported", path.stem, velocity_cfg)
                print("[SuperMimic] deployment files written to " + ", ".join(str(file) for file in exported))
            history = observations["actor"].unsqueeze(1).repeat(1, history_length, 1)
            velocity_command = observations["actor"].new_tensor(args.velocity_command).expand(adapter.num_envs, -1)

            for step in range(args.num_steps):
                actor_input = build_actor_obs_td(
                    {
                        "motion": extract_actor_goal(observations["goal"], frames),
                        "velocity": velocity_command,
                    },
                    history,
                )
                with torch.inference_mode():
                    action = actor(actor_input)
                observations, _, terminated, truncated, _ = adapter.step(action)
                history = advance_proprio_history(history, observations["actor"], terminated | truncated)
                if step % 100 == 0:
                    print(f"[SuperMimic] playback step={step}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
