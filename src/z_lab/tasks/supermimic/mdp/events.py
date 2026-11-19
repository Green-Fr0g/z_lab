"""Startup randomization for the SuperMimic dynamics context."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import ModelFlags, ShapeFlags

from isaaclab.utils.math import sample_uniform, sample_uniform_from_ranges

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def randomize_foot_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
    friction_range: tuple[float, float],
) -> None:
    """Apply one tangential friction coefficient to all foot colliders in each environment."""
    robot = env.scene[asset_cfg.name]
    physics = env.sim.physics_manager
    model = physics.get_model()
    view = robot._root_view
    body_ids = robot.map_body_ids_to_backend(asset_cfg.body_ids)
    shape_ids = [shape for body in body_ids for shape in view.body_shapes[body]]
    if not shape_ids:
        raise ValueError("No shapes found for SuperMimic's foot links.")
    indices = torch.tensor(shape_ids, device=env.device)
    flags = wp.to_torch(view.get_attribute("shape_flags", model)[:, 0])
    indices = indices[(flags[0, indices] & int(ShapeFlags.COLLIDE_SHAPES)) != 0]
    if indices.numel() == 0:
        raise ValueError("No collision shapes found for SuperMimic's foot links.")
    env_ids = slice(None) if env_ids is None else env_ids
    rows = env_ids if isinstance(env_ids, slice) else env_ids[:, None]
    count = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
    friction = wp.to_torch(view.get_attribute("shape_material_mu", model)[:, 0])
    friction[rows, indices] = sample_uniform(*friction_range, (count, 1), env.device)
    physics.add_model_change(ModelFlags.SHAPE_PROPERTIES)
    env.supermimic_foot_friction = friction[:, indices[:1]].clone()


def randomize_torso_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
) -> None:
    """Randomize the torso's link-frame CoM and store the realized offset [m]."""
    robot = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if not hasattr(env, "_supermimic_nominal_torso_com"):
        env._supermimic_nominal_torso_com = robot.data.body_com_pose_b.torch[:, body_ids].clone()
    nominal = env._supermimic_nominal_torso_com
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=robot.device)[env_ids]
    rows = slice(None) if env_ids is None else env_ids
    coms = nominal[rows].clone()
    coms[..., :3] += sample_uniform_from_ranges(com_range, ("x", "y", "z"), coms.shape[0], robot.device).unsqueeze(1)
    robot.set_coms_index(coms=coms, body_ids=body_ids, env_ids=env_ids)
    current = robot.data.body_com_pos_b.torch[:, body_ids]
    env.supermimic_com_offset = (current - nominal[..., :3]).squeeze(1)


def randomize_encoder_bias(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
    bias_range: tuple[float, float],
) -> None:
    """Store persistent encoder biases [rad] in articulation joint order."""
    robot = env.scene[asset_cfg.name]
    bias = torch.zeros((env.num_envs, len(robot.joint_names)), device=env.device)
    env_ids = slice(None) if env_ids is None else env_ids
    selected_bias = bias[env_ids]
    bias[env_ids] = sample_uniform(*bias_range, selected_bias.shape, env.device)
    env.supermimic_encoder_bias = bias
