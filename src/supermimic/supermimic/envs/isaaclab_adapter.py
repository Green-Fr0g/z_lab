"""Read SuperMimic transitions and randomization context from Isaac Lab tasks."""

from __future__ import annotations

from typing import Any

import torch

from supermimic.algorithms.representations import PROPRIO_OBS_TERMS
from supermimic.kinematics import RobotModel, load_robot_model

_TRACKING_REWARD_NAMES = (
    "motion_global_anchor_pos",
    "motion_global_anchor_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
)


class IsaacLabSuperMimicAdapter:
    """Adapt the registered manager-based task to the SuperMimic data contract."""

    def __init__(self, env: Any) -> None:
        self.env = env.unwrapped
        self.motion_command = self.env.command_manager.get_term("motion")
        self.action_term = self.env.action_manager.get_term("joint_pos")
        self.device = torch.device(self.env.device)
        self.num_envs = self.env.num_envs
        self.num_actions = self.action_term.action_dim
        self.joint_ids = self.action_term._joint_ids

    def load_robot_model(self) -> RobotModel:
        """Read the task's robot from the composed scene in action and tracking order."""
        from isaaclab.sim.utils.queries import find_first_matching_prim

        robot = self.motion_command.robot
        body_names = self.motion_command.cfg.body_names
        if robot.is_fixed_base or not body_names or body_names[0] != robot.body_names[0]:
            raise ValueError("The first tracked link must match the simulator's floating root link")
        robot_prim = find_first_matching_prim(robot.cfg.prim_path, stage=self.env.scene.stage)
        if robot_prim is None:
            raise ValueError(f"No robot prim found for {robot.cfg.prim_path!r}")
        return load_robot_model(robot_prim, self.action_term._joint_names, body_names)

    def observations(self) -> dict[str, torch.Tensor]:
        """Read the manager observations and snapshot the xyzw full state."""
        observations = self.env.obs_buf
        return {
            "full_state": observations["full_state"].clone(),
            "actor": observations["actor"],
            "goal": observations["goal"],
        }

    def validate_proprio_layout(self) -> int:
        """Check that the task exposes the proprioception synthesized by the model."""
        manager = self.env.observation_manager
        widths = (3, 3, self.num_actions, self.num_actions, self.num_actions)
        term_dims = [(width,) for width in widths]
        if (
            tuple(manager.active_terms["actor"]) != PROPRIO_OBS_TERMS
            or manager.group_obs_term_dim["actor"] != term_dims
        ):
            raise ValueError("Unexpected SuperMimic actor observation terms or widths")
        robot = self.motion_command.robot
        joint_names = list(self.action_term._joint_names)
        for name in ("joint_pos", "joint_vel"):
            asset_cfg = getattr(self.env.cfg.observations.actor, name).params["asset_cfg"]
            _, resolved_names = robot.find_joints(asset_cfg.joint_names, preserve_order=asset_cfg.preserve_order)
            if list(resolved_names) != joint_names:
                raise ValueError(f"Actor {name} joints must follow action order")
        motion_names = [self.motion_command.motion.joint_names[i] for i in self.motion_command.joint_indexes.tolist()]
        if motion_names != joint_names:
            raise ValueError("Motion goal joints must follow action order")
        return sum(widths)

    def motion_goal_window(self) -> tuple[int, int]:
        """Read the common frame count and stride from the task's goal terms."""
        goal = self.env.cfg.observations.goal
        names = ["command", "motion_anchor_pos_b", "motion_anchor_ori_b"]
        manager = self.env.observation_manager
        if manager.active_terms["goal"] != names:
            raise ValueError("Unexpected SuperMimic motion goal observation terms")
        params = goal.command.params
        frames, stride = params["num_frames"], params["frame_stride"]
        if frames <= 0 or stride <= 0:
            raise ValueError("Motion goal frame count and stride must be positive")
        for name in names[1:]:
            params = getattr(goal, name).params
            if (params["num_frames"], params["frame_stride"]) != (frames, stride):
                raise ValueError(f"Motion goal term {name!r} must use the command's frame count and stride")
        widths = [(frames * width,) for width in (2 * self.num_actions, 3, 6)]
        if manager.group_obs_term_dim["goal"] != widths:
            raise ValueError("Unexpected SuperMimic motion goal observation widths")
        return frames, stride

    def validate_full_state_layout(self, representation: str, width: int) -> None:
        """Check that the selected representation matches the task's full state."""
        if representation == "gencoord":
            expected = 13 + 2 * self.num_actions
        elif representation == "maxcoord":
            expected = 13 * len(self.motion_command.cfg.body_names)
        else:
            raise ValueError(f"Unknown SuperMimic representation {representation!r}")
        if width != expected:
            raise ValueError(f"Invalid {representation} full-state width {width}, expected {expected}")

    def motion_times(self) -> torch.Tensor:
        """Copy the reference frame indices before the command manager advances them."""
        return self.motion_command.time_steps.clone()

    def previous_actions(self) -> torch.Tensor:
        """Return the raw actions that produced the current observation."""
        return self.env.action_manager.action.clone()

    def sim_context(self, training_events: dict | None = None) -> torch.Tensor:
        """Normalize realized physics parameters, optionally in the training ranges."""
        events = self.env.cfg.events
        training_events = training_events or {}
        friction_range = training_events.get("foot_friction", {}).get("params", events.foot_friction.params)[
            "friction_range"
        ]
        com_ranges = training_events.get("base_com", {}).get("params", events.base_com.params)["com_range"]
        bias_range = training_events.get("encoder_bias", {}).get("params", events.encoder_bias.params)["bias_range"]
        friction = self.env.supermimic_foot_friction[:, :1]
        com_offset = self.env.supermimic_com_offset
        encoder_bias = self.env.supermimic_encoder_bias[:, self.joint_ids]
        context = torch.cat(
            (
                self._normalize(friction, friction_range),
                torch.cat(
                    [
                        self._normalize(com_offset[:, axis : axis + 1], com_ranges[key])
                        for axis, key in enumerate(("x", "y", "z"))
                    ],
                    dim=-1,
                ),
                self._normalize(encoder_bias, bias_range),
            ),
            dim=-1,
        )
        return context

    def encoder_bias_context(self, context_dim: int) -> tuple[slice, float, float]:
        """Return the columns and inverse normalization for encoder bias in policy order."""
        bounds = self.env.cfg.events.encoder_bias.params["bias_range"]
        scale, midpoint = self._normalization_parameters(bounds)
        return slice(context_dim - self.num_actions, context_dim), scale, midpoint

    def action_scale_offset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read action scaling and default joint target offset from the manager."""
        scale = self._expand_action_parameter(self.action_term._scale)
        offset = self._expand_action_parameter(self.action_term._offset)
        return scale, offset

    def default_joint_pos(self) -> torch.Tensor:
        """Read default positions in action joint order."""
        return self.env.scene["robot"].data.default_joint_pos.torch[0, self.joint_ids].clone()

    def default_joint_vel(self) -> torch.Tensor:
        """Read default velocities in action joint order."""
        return self.env.scene["robot"].data.default_joint_vel.torch[0, self.joint_ids]

    def soft_joint_pos_limits(self) -> torch.Tensor:
        """Read soft joint limits in action joint order."""
        return self.env.scene["robot"].data.soft_joint_pos_limits.torch[0, self.joint_ids].clone()

    def reward_weights(self) -> dict[str, float]:
        """Return task-configured tracking and regularization weights."""
        rewards = self.env.cfg.rewards
        names = (*_TRACKING_REWARD_NAMES, "action_rate_l2", "joint_limit")
        return {name: float(getattr(rewards, name).weight) for name in names}

    def reward_stds(self) -> dict[str, float]:
        """Return the tracking reward standard deviations from the task config."""
        rewards = self.env.cfg.rewards
        return {name: float(getattr(rewards, name).params["std"]) for name in _TRACKING_REWARD_NAMES}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step simulation and return observations, rewards, and termination causes."""
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        self.motion_command = self.env.command_manager.get_term("motion")
        return (
            self.observations(),
            rewards,
            terminated,
            truncated,
            extras,
        )

    @classmethod
    def _normalize(cls, values: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
        scale, midpoint = cls._normalization_parameters(bounds)
        return (values - midpoint) / scale

    @staticmethod
    def _normalization_parameters(bounds: tuple[float, float]) -> tuple[float, float]:
        lower, upper = bounds
        midpoint = (lower + upper) * 0.5
        half_width = (upper - lower) * 0.5
        scale = half_width if half_width > 1e-3 else 1.0
        return scale, midpoint

    def _expand_action_parameter(self, value: float | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            values = value[0] if value.ndim > 1 else value
            return values.to(device=self.device, dtype=torch.float32).clone()
        return torch.full((self.num_actions,), float(value), device=self.device)
