"""Play a SuperMimic actor from a local checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--task-encoder", choices=("motion", "velocity"), default="motion")
    parser.add_argument("--velocity-command", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--physics", default="newton_mjwarp", help="Isaac Lab physics preset for the task.")
    add_launcher_args(parser)
    args = parser.parse_args()

    import gymnasium as gym
    from supermimic.algorithms.models import MultiEncoderAE, TaskSpec
    from supermimic.algorithms.representations import extract_actor_goal
    from supermimic.envs.isaaclab_adapter import IsaacLabSuperMimicAdapter
    from supermimic.tasks import advance_proprio_history, build_actor_obs_td

    from isaaclab_tasks.utils import parse_env_cfg

    import z_lab.tasks  # noqa: F401

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("actor_obs_mode") != "proprio":
        raise ValueError("Playback requires a checkpoint trained with proprioception. Retrain the rep_state actor.")
    saved_config = checkpoint["config"]
    history_length = saved_config["history_length"]
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        overrides=(f"physics={args.physics}",),
    )
    env_cfg.play_mode()
    if args.motion_file is not None:
        env_cfg.commands.motion.motion_file = str(args.motion_file)

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            env.reset()
            adapter = IsaacLabSuperMimicAdapter(env)
            observations = adapter.observations()
            obs_dim = adapter.validate_proprio_layout()
            if list(adapter.action_term._joint_names) != checkpoint["joint_names"]:
                raise ValueError("Checkpoint joints must match the playback task's action order")
            frames = saved_config["num_goal_frames"]
            goal_dim = extract_actor_goal(observations["goal"], frames).shape[-1]
            task_names = checkpoint.get("task_names", ["motion"])
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
