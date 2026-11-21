"""Train a SuperMimic world model and actor on an Isaac Lab task."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-iterations", type=int, default=1_000_000)
    parser.add_argument("--motion-file", type=Path)
    parser.add_argument("--sampling-mode", choices=("return", "uniform", "start"), default="return")
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--replay-ratio", type=float, default=16.0)
    parser.add_argument("--wm-batch-size", type=int, default=2048)
    parser.add_argument("--wm-num-batches", type=int, default=10)
    parser.add_argument("--wm-horizon", type=int, default=8)
    parser.add_argument("--policy-batch-size", type=int, default=4096)
    parser.add_argument("--policy-horizon", type=int, default=32)
    parser.add_argument("--policy-learner", choices=("actor-critic", "bptt"), default="actor-critic")
    parser.add_argument("--history-length", type=int, default=1)
    parser.add_argument("--motion-slot-fraction", type=float, default=0.5)
    parser.add_argument("--run-dir", type=Path, default=Path("logs/supermimic"))
    parser.add_argument("--physics", default="newton_mjwarp", help="Isaac Lab physics preset for the task.")
    add_launcher_args(parser)
    args = parser.parse_args()
    if not 0.0 < args.motion_slot_fraction < 1.0:
        parser.error("--motion-slot-fraction must be between 0 and 1, exclusive")

    import gymnasium as gym
    from supermimic.algorithms.learners.policy_learner import ActorCriticPolicyLearnerCfg, BpttPolicyLearnerCfg
    from supermimic.algorithms.learners.world_model_learner import WorldModelLearnerCfg
    from supermimic.runners.runner import SuperMimicRunner, SuperMimicRunnerCfg

    from isaaclab_tasks.utils import parse_env_cfg

    import z_lab.tasks  # noqa: F401

    if "GenCoord" in args.task:
        representation = "gencoord"
    elif "MaxCoord" in args.task:
        representation = "maxcoord"
    else:
        raise ValueError("SuperMimic tasks must select GenCoord or MaxCoord")
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        overrides=(f"physics={args.physics}",),
    )
    if args.motion_file is not None:
        env_cfg.commands.motion.motion_file = str(args.motion_file)
    env_cfg.commands.motion.sampling_mode = "external" if args.sampling_mode == "return" else args.sampling_mode

    policy_cfg_cls = ActorCriticPolicyLearnerCfg if args.policy_learner == "actor-critic" else BpttPolicyLearnerCfg

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            cfg = SuperMimicRunnerCfg(
                max_iterations=args.max_iterations,
                representation=representation,
                replay_buffer_size=args.buffer_size,
                replay_ratio=args.replay_ratio,
                sampling_mode=args.sampling_mode,
                history_length=args.history_length,
                slot_mix={
                    "motion": args.motion_slot_fraction,
                    "velocity": 1.0 - args.motion_slot_fraction,
                },
                run_dir=args.run_dir,
                world_model=WorldModelLearnerCfg(
                    batch_size=args.wm_batch_size,
                    num_batches=args.wm_num_batches,
                    horizon=args.wm_horizon,
                ),
                policy=policy_cfg_cls(
                    batch_size=args.policy_batch_size,
                    horizon=args.policy_horizon,
                ),
            )
            runner = SuperMimicRunner(env, cfg)
            output_dir = runner.run()
            print(f"[SuperMimic] checkpoints written to {output_dir}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
