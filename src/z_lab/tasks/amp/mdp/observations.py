"""Observations used by the AMP discriminator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms

from .motion_loader import MotionResetManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _value(value):
    """Return the torch view for Isaac Lab 3.0 proxy arrays."""
    return value.torch if hasattr(value, "torch") else value


def _get_body_views(env: ManagerBasedRLEnv, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg):
    asset = env.scene[anchor_cfg.name]
    body_pos_w_all = _value(asset.data.body_link_pos_w)
    body_quat_w_all = _value(asset.data.body_link_quat_w)
    anchor_pos_w = body_pos_w_all[:, anchor_cfg.body_ids[0]]
    anchor_quat_w = body_quat_w_all[:, anchor_cfg.body_ids[0]]
    body_pos_w = body_pos_w_all[:, body_cfg.body_ids]
    body_quat_w = body_quat_w_all[:, body_cfg.body_ids]
    return asset, anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w


def robot_body_pos_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return selected body positions in the configured anchor frame."""
    _, anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w = _get_body_views(env, anchor_cfg, body_cfg)
    num_bodies = body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return relative body orientation as the first two rotation-matrix columns."""
    _, anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w = _get_body_views(env, anchor_cfg, body_cfg)
    num_bodies = body_pos_w.shape[1]
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return matrix_from_quat(ori_b)[..., :2].reshape(env.num_envs, -1)


def robot_body_lin_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return selected body linear velocities in each body frame."""
    asset = env.scene[anchor_cfg.name]
    body_lin_vel_w = _value(asset.data.body_link_lin_vel_w)[:, body_cfg.body_ids]
    body_quat_w = _value(asset.data.body_link_quat_w)[:, body_cfg.body_ids]
    body_lin_vel_b = quat_apply_inverse(body_quat_w, body_lin_vel_w)
    return body_lin_vel_b.reshape(env.num_envs, -1)


def robot_body_ang_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return selected body angular velocities in each body frame."""
    asset = env.scene[anchor_cfg.name]
    body_ang_vel_w = _value(asset.data.body_link_ang_vel_w)[:, body_cfg.body_ids]
    body_quat_w = _value(asset.data.body_link_quat_w)[:, body_cfg.body_ids]
    body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)
    return body_ang_vel_b.reshape(env.num_envs, -1)


def robot_anchor_lin_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return the anchor body's linear velocity in the anchor frame."""
    asset = env.scene[anchor_cfg.name]
    body_ids = anchor_cfg.body_ids[0]
    body_lin_vel_w = _value(asset.data.body_link_lin_vel_w)[:, body_ids]
    body_quat_w = _value(asset.data.body_link_quat_w)[:, body_ids]
    return quat_apply_inverse(body_quat_w, body_lin_vel_w)


def robot_anchor_ang_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return the anchor body's angular velocity in the anchor frame."""
    asset = env.scene[anchor_cfg.name]
    body_ids = anchor_cfg.body_ids[0]
    body_ang_vel_w = _value(asset.data.body_link_ang_vel_w)[:, body_ids]
    body_quat_w = _value(asset.data.body_link_quat_w)[:, body_ids]
    return quat_apply_inverse(body_quat_w, body_ang_vel_w)


def robot_body_projected_gravity_b(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return the world gravity vector projected into each selected body frame."""
    asset = env.scene[body_cfg.name]
    body_quat_w = _value(asset.data.body_link_quat_w)[:, body_cfg.body_ids]
    gravity_w = _value(asset.data.GRAVITY_VEC_W)
    if gravity_w.ndim == 1:
        gravity_w = gravity_w.expand(env.num_envs, -1)
    gravity_w = gravity_w[:, None, :].expand(-1, body_quat_w.shape[1], -1)
    return quat_apply_inverse(body_quat_w, gravity_w).reshape(env.num_envs, -1)


def robot_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
    """Return positions of the selected robot joints in radians."""
    return _value(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids]


def robot_joint_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
    """Return velocities of the selected robot joints in radians per second."""
    return _value(env.scene[asset_cfg.name].data.joint_vel)[:, asset_cfg.joint_ids]


def _get_motion_views(env: ManagerBasedRLEnv, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg):
    """Get online expert body views using the same semantic names as the robot terms."""
    manager = MotionResetManager.get()
    if manager._reference:
        return manager.reference_body_data(
            anchor_name=anchor_cfg.body_names[0],
            body_names=body_cfg.body_names,
        )

    # ObservationManager probes term shapes before startup events run. Return a
    # correctly shaped neutral reference for that probe; the reset event fills
    # the online reference with the sampled motion before the first step.
    num_bodies = len(body_cfg.body_names)
    anchor_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    anchor_quat_w = torch.zeros((env.num_envs, 4), device=env.device)
    anchor_quat_w[:, 0] = 1.0
    body_pos_w = torch.zeros((env.num_envs, num_bodies, 3), device=env.device)
    body_quat_w = torch.zeros((env.num_envs, num_bodies, 4), device=env.device)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = torch.zeros_like(body_pos_w)
    body_ang_vel_w = torch.zeros_like(body_pos_w)
    return anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w


def motion_body_pos_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return online expert body positions in the configured anchor frame."""
    anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w, _, _ = _get_motion_views(env, anchor_cfg, body_cfg)
    num_bodies = body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def motion_body_ori_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return online expert relative orientations as six-dimensional matrices."""
    anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w, _, _ = _get_motion_views(env, anchor_cfg, body_cfg)
    num_bodies = body_pos_w.shape[1]
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return matrix_from_quat(ori_b)[..., :2].reshape(env.num_envs, -1)


def motion_body_lin_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return online expert body-frame linear velocities."""
    _, _, _, body_quat_w, body_lin_vel_w, _ = _get_motion_views(env, anchor_cfg, body_cfg)
    return quat_apply_inverse(body_quat_w, body_lin_vel_w).reshape(env.num_envs, -1)


def motion_body_ang_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return online expert body-frame angular velocities."""
    _, _, _, body_quat_w, _, body_ang_vel_w = _get_motion_views(env, anchor_cfg, body_cfg)
    return quat_apply_inverse(body_quat_w, body_ang_vel_w).reshape(env.num_envs, -1)


def motion_anchor_lin_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return the online reference anchor body's linear velocity in its frame."""
    _, _, _, anchor_quat_w, anchor_lin_vel_w, _ = _get_motion_views(env, anchor_cfg, anchor_cfg)
    return quat_apply_inverse(anchor_quat_w[:, 0], anchor_lin_vel_w[:, 0])


def motion_anchor_ang_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return the online reference anchor body's angular velocity in its frame."""
    _, _, _, anchor_quat_w, _, anchor_ang_vel_w = _get_motion_views(env, anchor_cfg, anchor_cfg)
    return quat_apply_inverse(anchor_quat_w[:, 0], anchor_ang_vel_w[:, 0])


def motion_body_projected_gravity_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return world gravity projected into selected online reference body frames."""
    _, _, _, body_quat_w, _, _ = _get_motion_views(env, anchor_cfg, body_cfg)
    gravity_w = _value(env.scene[body_cfg.name].data.GRAVITY_VEC_W)
    if gravity_w.ndim == 1:
        gravity_w = gravity_w.expand(env.num_envs, -1)
    gravity_w = gravity_w[:, None, :].expand(-1, body_quat_w.shape[1], -1)
    return quat_apply_inverse(body_quat_w, gravity_w).reshape(env.num_envs, -1)


def motion_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
    """Return positions of selected joints in the current online reference frame."""
    manager = MotionResetManager.get()
    if manager._clip_ids is None or manager._frame_ids is None or not manager._cache_key:
        raise RuntimeError("AMP reference observations were requested before motion reset.")

    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    joint_names = tuple(asset.joint_names)
    result = torch.empty((env.num_envs, len(joint_ids)), device=env.device, dtype=torch.float32)
    recovery = manager._recovery_active
    clip_sets = (
        (False, manager._clips[manager._cache_key]),
        (True, manager._recovery_clips[manager._cache_key]),
    )
    for is_recovery, clips in clip_sets:
        selected_envs = torch.nonzero(recovery == is_recovery, as_tuple=False).flatten()
        if selected_envs.numel() == 0:
            continue
        for clip_id in manager._clip_ids[selected_envs].unique().tolist():
            env_ids = selected_envs[manager._clip_ids[selected_envs] == clip_id]
            clip = clips[clip_id]
            names = {name: index for index, name in enumerate(clip.dof_names)}
            source_ids = torch.tensor([names[name] for name in joint_names], device=env.device)
            frame_ids = manager._frame_ids[env_ids]
            result[env_ids] = clip.dof_pos[frame_ids][:, source_ids][:, joint_ids]
    return result


def motion_joint_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
    """Return velocities of selected joints in the current online reference frame."""
    manager = MotionResetManager.get()
    if manager._clip_ids is None or manager._frame_ids is None or not manager._cache_key:
        raise RuntimeError("AMP reference observations were requested before motion reset.")
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    joint_names = tuple(asset.joint_names)
    result = torch.empty((env.num_envs, len(joint_ids)), device=env.device, dtype=torch.float32)
    clip_sets = (
        (False, manager._clips[manager._cache_key]),
        (True, manager._recovery_clips[manager._cache_key]),
    )
    for is_recovery, clips in clip_sets:
        selected_envs = torch.nonzero(manager._recovery_active == is_recovery, as_tuple=False).flatten()
        if selected_envs.numel() == 0:
            continue
        for clip_id in manager._clip_ids[selected_envs].unique().tolist():
            env_ids = selected_envs[manager._clip_ids[selected_envs] == clip_id]
            clip = clips[clip_id]
            names = {name: index for index, name in enumerate(clip.dof_names)}
            source_ids = torch.tensor([names[name] for name in joint_names], device=env.device)
            frame_ids = manager._frame_ids[env_ids]
            result[env_ids] = clip.dof_vel[frame_ids][:, source_ids][:, joint_ids]
    return result
