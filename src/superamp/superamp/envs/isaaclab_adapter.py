"""Read SuperAMP transitions and randomization context from Isaac Lab tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from superamp.kinematics import RobotModel, load_robot_model
from superamp.representations import PROPRIO_OBS_TERMS
from superamp.storage import ExpertTransitions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_EXP_REWARD_NAMES = ("track_lin_vel_xy", "track_ang_vel_z", "body_ang_vel_xy_l2")


class IsaacLabSuperAmpAdapter:
    """Adapt the registered manager-based task to the SuperAMP data contract."""

    def __init__(self, env: ManagerBasedRLEnv) -> None:
        self.env = env.unwrapped
        self.action_term = self.env.action_manager.get_term("joint_pos")
        self.device = torch.device(self.env.device)
        self.num_envs = self.env.num_envs
        self.num_actions = self.action_term.action_dim
        self.joint_ids = self.action_term._joint_ids
        self.robot = self.env.scene[self.action_term.cfg.asset_name]
        if not self.env.cfg.compute_final_obs:
            raise ValueError("SuperAMP collection requires compute_final_obs")

        self.observation_dims: dict[str, int] = {}
        group_dims = self.env.observation_manager.group_obs_dim
        for name in ("actor", "full_state", "goal"):
            dims = group_dims.get(name)
            if not isinstance(dims, tuple) or len(dims) != 1:
                raise ValueError(f"SuperAMP requires a flat {name!r} observation group")
            self.observation_dims[name] = dims[0]

    def load_robot_model(self) -> RobotModel:
        """Read the task's robot from the composed scene in action and tracking order."""
        from isaaclab.sim.utils.queries import find_first_matching_prim

        robot = self.robot
        body_names = self.env.cfg.full_state_body_names
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
            raise ValueError("Unexpected SuperAMP actor observation terms or widths")
        robot = self.robot
        joint_names = list(self.action_term._joint_names)
        for name in ("joint_pos", "joint_vel"):
            asset_cfg = getattr(self.env.cfg.observations.actor, name).params["asset_cfg"]
            _, resolved_names = robot.find_joints(asset_cfg.joint_names, preserve_order=asset_cfg.preserve_order)
            if list(resolved_names) != joint_names:
                raise ValueError(f"Actor {name} joints must follow action order")
        return sum(widths)

    def validate_full_state_layout(self, representation: str, width: int) -> None:
        """Check that the selected representation matches the task's full state."""
        if representation == "gencoord":
            expected = 13 + 2 * self.num_actions
            asset_cfg = self.env.cfg.observations.full_state.state.params["asset_cfg"]
            _, names = self.robot.find_joints(asset_cfg.joint_names, preserve_order=asset_cfg.preserve_order)
            if list(names) != list(self.action_term._joint_names):
                raise ValueError("GenCoord state joints must follow action order")
        elif representation == "maxcoord":
            expected = 13 * len(self.env.cfg.full_state_body_names)
            asset_cfg = self.env.cfg.observations.full_state.state.params["asset_cfg"]
            _, names = self.robot.find_bodies(asset_cfg.body_names, preserve_order=asset_cfg.preserve_order)
            if list(names) != list(self.env.cfg.full_state_body_names):
                raise ValueError("MaxCoord state links must follow the configured body order")
        else:
            raise ValueError(f"Unknown SuperAMP representation {representation!r}")
        if width != expected:
            raise ValueError(f"Invalid {representation} full-state width {width}, expected {expected}")

    def previous_actions(self) -> torch.Tensor:
        """Return the raw actions that produced the current observation."""
        return self.env.action_manager.action.clone()

    def sim_context(self) -> torch.Tensor:
        """Normalize realized physics parameters in the configured randomization ranges."""
        events = self.env.cfg.events
        friction_range = events.foot_friction.params["friction_range"]
        com_ranges = events.base_com.params["com_range"]
        bias_range = events.encoder_bias.params["bias_range"]
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
        return self.robot.data.default_joint_pos.torch[0, self.joint_ids].clone()

    def default_joint_vel(self) -> torch.Tensor:
        """Read default velocities in action joint order."""
        return self.robot.data.default_joint_vel.torch[0, self.joint_ids]

    def soft_joint_pos_limits(self) -> torch.Tensor:
        """Read soft joint limits in action joint order."""
        return self.robot.data.soft_joint_pos_limits.torch[0, self.joint_ids].clone()

    def reward_weights(self) -> dict[str, float]:
        """Return task-configured tracking and regularization weights."""
        manager = self.env.reward_manager
        names = (*_EXP_REWARD_NAMES, "joint_acc_l2", "action_rate_l2", "joint_pos_limits")
        return {name: float(manager.get_term_cfg(name).weight) for name in names}

    def reward_params(self) -> dict[str, dict]:
        """Resolve task reward parameters in world-model body order."""
        manager = self.env.reward_manager
        params = {name: {"std": float(manager.get_term_cfg(name).params["std"])} for name in _EXP_REWARD_NAMES}
        body_cfg = manager.get_term_cfg("body_ang_vel_xy_l2").params["body_cfg"]
        body_name = self.robot.body_names[body_cfg.body_ids[0]]
        params["body_ang_vel_xy_l2"]["body_index"] = self.env.cfg.full_state_body_names.index(body_name)
        return params

    def expert_transitions(self, *, num_frames: int = 2, frame_stride: int = 1) -> ExpertTransitions:
        """Load discriminator examples using the task's expert paths and AMP link order."""
        reset = self.env.cfg.events.reset_from_expert.params
        return ExpertTransitions(
            reset["motion_file"],
            self.env.cfg.amp_body_names,
            self.env.cfg.anchor_link_name,
            control_dt=self.env.step_dt,
            num_frames=num_frames,
            frame_stride=frame_stride,
            device=self.device,
            exclude_motion_names=reset["exclude_motion_names"],
        )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step simulation and return observations, rewards, and termination causes."""
        _, rewards, terminated, truncated, extras = self.env.step(actions)
        return self.observations(), rewards, terminated, truncated, extras

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
