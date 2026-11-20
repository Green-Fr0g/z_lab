"""Task goal contracts and command construction."""

from .task_spec import VelocityTaskCfg, advance_proprio_history, build_actor_obs_td, velocity_goal_from_motion

__all__ = ["VelocityTaskCfg", "advance_proprio_history", "build_actor_obs_td", "velocity_goal_from_motion"]
