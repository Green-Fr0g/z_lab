"""Train SuperAMP on a registered Isaac Lab velocity-control task."""

from __future__ import annotations

import argparse
from datetime import datetime

import gymnasium as gym
import torch
from cli_args import add_common_args, checkpoint_path, parse_task_configs
from superamp.learners import (
    ActorCriticPolicyLearnerCfg,
    BpttPolicyLearnerCfg,
    DiscriminatorLearnerCfg,
    WorldModelLearnerCfg,
)
from superamp.runners import ImaginedSuperAmpRunner, SuperAmpRunner, SuperAmpRunnerCfg

from isaaclab.app import launch_simulation

from isaaclab_rl.entrypoints.common import dump_train_configs, write_run_manifest

from z_lab.tasks.superamp.rl_cfg import SuperAmpRunnerCfg as AgentCfg


def build_runner_cfg(agent_cfg: AgentCfg) -> SuperAmpRunnerCfg:
    """Convert a registered configuration to the learner's dataclasses."""
    values = agent_cfg.to_dict()
    for name in ("experiment_name", "run_name", "seed"):
        del values[name]
    policy = values.pop("policy")
    kind = policy.pop("kind")
    if kind == "actor-critic":
        policy_class = ActorCriticPolicyLearnerCfg
    elif kind == "bptt":
        policy_class = BpttPolicyLearnerCfg
        for name in ("lam", "critic_updates", "critic_mini_batches"):
            del policy[name]
    else:
        raise ValueError(f"Unknown SuperAMP policy learner {kind!r}")
    if policy["gamma"] is None:
        del policy["gamma"]
    values["policy"] = policy_class(**policy)
    values["world_model"] = WorldModelLearnerCfg(**values["world_model"])
    values["discriminator"] = DiscriminatorLearnerCfg(**values["discriminator"])
    return SuperAmpRunnerCfg(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--max_iterations", "--max-iterations", type=int)
    parser.add_argument("--run_name", "--run-name")
    parser.add_argument("--replay-ratio", type=float)
    parser.add_argument("--world-model-checkpoint", help="Standalone world-model checkpoint, or 'latest'.")
    parser.add_argument("--source", choices=("true", "imagine"), default="true", help="Source of AMP policy samples.")
    args, env_cfg, agent_cfg = parse_task_configs(parser)
    if args.checkpoint and args.world_model_checkpoint:
        parser.error("--checkpoint and --world-model-checkpoint are mutually exclusive")
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name
    if args.source == "imagine":
        agent_cfg.run_name = f"{agent_cfg.run_name}_imagine" if agent_cfg.run_name else "imagine"
    if args.replay_ratio is not None:
        agent_cfg.replay_ratio = args.replay_ratio
    cfg = build_runner_cfg(agent_cfg)
    agent_cfg.policy.gamma = cfg.policy.gamma
    log_root = args.run_dir / agent_cfg.experiment_name
    resume_path = checkpoint_path(args, log_root, args.checkpoint) if args.checkpoint else None
    world_model_path = (
        checkpoint_path(args, log_root, args.world_model_checkpoint, world_model=True)
        if args.world_model_checkpoint
        else None
    )
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = (log_root / run_name).resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(log_dir)

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            checkpoint = torch.load(resume_path, map_location=env.unwrapped.device) if resume_path is not None else None
            runner_class = ImaginedSuperAmpRunner if args.source == "imagine" else SuperAmpRunner
            runner = runner_class(env, cfg)
            metadata = {"agent": args.agent, "amp_policy_source": args.source}
            if checkpoint is not None:
                runner.load(checkpoint)
                del checkpoint
            elif world_model_path is not None:
                runner.load_world_model(torch.load(world_model_path, map_location="cpu"))
                metadata["world_model_checkpoint"] = str(world_model_path)
            agent_cfg.seed = env.unwrapped.cfg.seed
            dump_train_configs(str(log_dir), env.unwrapped.cfg, agent_cfg)
            write_run_manifest(str(log_dir), library="superamp", task=args.task, metadata=metadata)
            runner.run(log_dir, {"task": args.task, **metadata})
            print(f"[SuperAMP] checkpoints written to {log_dir}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
