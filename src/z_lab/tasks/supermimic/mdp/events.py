"""Startup randomization for the SuperMimic dynamics context."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp import randomize_rigid_body_material
from isaaclab.utils.math import sample_uniform, sample_uniform_from_ranges
from isaaclab_newton.physics import NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg, SceneEntityCfg


class randomize_foot_friction(randomize_rigid_body_material):
    """Use Isaac Lab's material bindings with a shared friction sample per environment."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        friction_range = cfg.params["friction_range"]
        super().__init__(
            cfg.replace(
                params={
                    "asset_cfg": cfg.params["asset_cfg"],
                    "static_friction_range": friction_range,
                    "dynamic_friction_range": friction_range,
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 64,
                    "make_consistent": True,
                }
            ),
            env,
        )
        env.supermimic_foot_friction = torch.zeros((env.num_envs, 1), device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        friction_range: tuple[float, float],
    ) -> None:
        ids = torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
        if ids.numel() == 0:
            return
        physics = env.sim.cfg.physics
        if isinstance(physics, NewtonCfg):
            values = self._set_newton_friction(ids, friction_range)
        elif isinstance(physics, (PhysxCfg, OvPhysxCfg)):
            buckets = self._impl.material_buckets[:, 0]
            values = buckets[torch.randint(len(buckets), (ids.numel(),), device=buckets.device)]
            if isinstance(physics, OvPhysxCfg):
                self._set_ovphysx_friction(ids, values)
            else:
                self._set_physx_friction(ids, values)
        else:
            raise NotImplementedError(f"Shared foot friction is unavailable for {type(physics).__name__}")
        env.supermimic_foot_friction[ids] = values.to(env.device).unsqueeze(-1)

    def _set_newton_friction(self, ids: torch.Tensor, limits: tuple[float, float]) -> torch.Tensor:
        from newton import ModelFlags, ShapeFlags
        from newton.solvers import SolverKamino

        impl = self._impl
        physics = impl._newton_manager
        friction = wp.to_torch(impl._friction_binding)
        model = physics.get_model()
        flags = wp.to_torch(self.asset.root_view.get_attribute("shape_flags", model)[:, 0])
        shapes = impl._shape_indices.to(friction.device)
        shapes = shapes[(flags[0, shapes] & int(ShapeFlags.COLLIDE_SHAPES)) != 0]
        if not shapes.numel():
            raise ValueError("No collision shapes found for SuperMimic's foot links.")
        if isinstance(physics._solver, SolverKamino):
            # Kamino shares material groups across environments.
            value = sample_uniform(*limits, (1,), friction.device)
            friction[:, shapes] = value
            self._env.supermimic_foot_friction[:] = value
            values = value.expand(ids.numel())
        else:
            values = sample_uniform(*limits, (ids.numel(),), friction.device)
            friction[ids[:, None], shapes] = values[:, None]
        physics.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        return values

    def _set_physx_friction(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        impl = self._impl
        view = self.asset.root_view
        materials = wp.to_torch(view.get_material_properties())
        rows = ids.to(materials.device)
        values = values.to(materials.device)[:, None, None]
        body_ids = impl._backend_body_ids
        if body_ids is None:
            materials[rows, :, :2] = values
        else:
            for body in body_ids:
                start = sum(impl.num_shapes_per_body[:body])
                end = start + impl.num_shapes_per_body[body]
                materials[rows, start:end, :2] = values
        indices = wp.from_torch(rows.to(torch.int32), dtype=wp.int32)
        view.set_material_properties(wp.from_torch(materials, dtype=wp.float32), indices)

    def _set_ovphysx_friction(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        impl = self._impl
        view = impl._material_view
        materials = wp.to_torch(view.get_attribute(impl._material_type))
        rows_by_env = impl._material_rows_by_env[ids.cpu()].to(materials.device)
        materials[rows_by_env.flatten(), :, :2] = values.to(materials.device).repeat_interleave(rows_by_env.shape[1])[
            :, None, None
        ]
        indices = wp.from_torch(rows_by_env.flatten().to(torch.int32), dtype=wp.int32)
        view.set_attribute(
            impl._material_type, wp.from_torch(materials.contiguous(), dtype=wp.float32), indices=indices
        )


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
