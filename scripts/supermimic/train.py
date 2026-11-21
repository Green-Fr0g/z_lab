"""Train a SuperMimic world model and actor on a registered Isaac Lab task."""

from __future__ import annotations

import argparse
from datetime import datetime

import gymnasium as gym
import torch
from cli_args import add_common_args, checkpoint_path, parse_task_configs
from supermimic.algorithms.learners.policy_learner import ActorCriticPolicyLearnerCfg, BpttPolicyLearnerCfg
from supermimic.algorithms.learners.world_model_learner import WorldModelLearnerCfg
from supermimic.runners.reset_sampling import ReturnSamplerCfg
from supermimic.runners.runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.tasks import VelocityTaskCfg

from isaaclab.app import launch_simulation

from isaaclab_rl.entrypoints.common import dump_train_configs, write_run_manifest

from z_lab.tasks.supermimic.rl_cfg import SuperMimicRunnerCfg as AgentCfg


def build_runner_cfg(agent_cfg: AgentCfg) -> SuperMimicRunnerCfg:
    """Convert the task-facing configuration to algorithm-side types."""
    values = agent_cfg.to_dict()
    for key in ("experiment_name", "run_name", "seed"):
        del values[key]
    policy = values.pop("policy")
    kind = policy.pop("kind")
    if kind == "actor-critic":
        policy_class = ActorCriticPolicyLearnerCfg
    elif kind == "bptt":
        policy_class = BpttPolicyLearnerCfg
        for key in ("lam", "critic_updates", "critic_mini_batches"):
            del policy[key]
    else:
        raise ValueError(f"Unknown SuperMimic policy learner {kind!r}")
    return SuperMimicRunnerCfg(
        **{
            **values,
            "policy": policy_class(**policy),
            "world_model": WorldModelLearnerCfg(**values["world_model"]),
            "velocity": VelocityTaskCfg(**values["velocity"]),
            "return_sampler": ReturnSamplerCfg(**values["return_sampler"]),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--max_iterations", "--max-iterations", type=int)
    parser.add_argument("--run_name", "--run-name")
    parser.add_argument("--sampling-mode", choices=("return", "uniform", "start"))
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--replay-ratio", type=float)
    parser.add_argument("--wm-batch-size", type=int)
    parser.add_argument("--wm-num-batches", type=int)
    parser.add_argument("--wm-horizon", type=int)
    parser.add_argument("--policy-batch-size", type=int)
    parser.add_argument("--policy-horizon", type=int)
    parser.add_argument("--policy-learner", choices=("actor-critic", "bptt"))
    parser.add_argument("--history-length", type=int)
    parser.add_argument("--motion-slot-fraction", type=float)
    args, env_cfg, agent_cfg = parse_task_configs(parser)
    for obj, field, override in (
        (agent_cfg, "max_iterations", args.max_iterations),
        (agent_cfg, "run_name", args.run_name),
        (agent_cfg, "sampling_mode", args.sampling_mode),
        (agent_cfg, "replay_buffer_size", args.buffer_size),
        (agent_cfg, "replay_ratio", args.replay_ratio),
        (agent_cfg, "history_length", args.history_length),
        (agent_cfg.world_model, "batch_size", args.wm_batch_size),
        (agent_cfg.world_model, "num_batches", args.wm_num_batches),
        (agent_cfg.world_model, "horizon", args.wm_horizon),
        (agent_cfg.policy, "batch_size", args.policy_batch_size),
        (agent_cfg.policy, "horizon", args.policy_horizon),
        (agent_cfg.policy, "kind", args.policy_learner),
    ):
        if override is not None:
            setattr(obj, field, override)
    if args.motion_slot_fraction is not None:
        agent_cfg.slot_mix = {"motion": args.motion_slot_fraction, "velocity": 1.0 - args.motion_slot_fraction}
    if agent_cfg.policy.gamma is None:
        agent_cfg.policy.gamma = 0.99 if agent_cfg.policy.kind == "actor-critic" else 1.0
    if agent_cfg.sampling_mode not in ("return", "uniform", "start"):
        parser.error("sampling_mode must be 'return', 'uniform' or 'start'")
    if min(agent_cfg.max_iterations, agent_cfg.log_interval, agent_cfg.save_interval) <= 0:
        parser.error("max_iterations, log_interval and save_interval must be positive")
    if agent_cfg.replay_buffer_size < max(2, agent_cfg.world_model.horizon, agent_cfg.history_length):
        parser.error("Replay capacity must cover the world-model horizon and actor history")

    cfg = build_runner_cfg(agent_cfg)
    env_cfg.commands.motion.sampling_mode = "external" if cfg.sampling_mode == "return" else cfg.sampling_mode
    log_root = args.run_dir / agent_cfg.experiment_name
    resume_path = checkpoint_path(args, log_root, args.checkpoint) if args.checkpoint else None
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = (log_root / run_name).resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(log_dir)

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            runner = SuperMimicRunner(env, cfg)
            if resume_path is not None:
                checkpoint = torch.load(resume_path, map_location=runner.adapter.device, weights_only=False)
                runner.load(checkpoint)
                print(f"[SuperMimic] resumed from {resume_path}")
            env_cfg = env.unwrapped.cfg
            agent_cfg.seed = env_cfg.seed
            infos = {"task": args.task, "agent": args.agent}
            dump_train_configs(str(log_dir), env_cfg, agent_cfg)
            write_run_manifest(str(log_dir), library="supermimic", task=args.task, metadata={"agent": args.agent})
            runner.run(log_dir, infos)
            print(f"[SuperMimic] checkpoints written to {log_dir}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
