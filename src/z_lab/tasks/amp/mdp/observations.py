"""AMP discriminator observations matching AMP_mjlab's body-frame features."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _body_data(env: ManagerBasedEnv, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg):
    asset = env.scene[body_cfg.name]
    anchor_pos = asset.data.body_link_pos_w.torch[:, anchor_cfg.body_ids[0]]
    anchor_quat = asset.data.body_link_quat_w.torch[:, anchor_cfg.body_ids[0]]
    body_pos = asset.data.body_link_pos_w.torch[:, body_cfg.body_ids]
    body_quat = asset.data.body_link_quat_w.torch[:, body_cfg.body_ids]
    return asset, anchor_pos, anchor_quat, body_pos, body_quat


def robot_body_pos_b(env: ManagerBasedEnv, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg):
    _, anchor_pos, anchor_quat, body_pos, body_quat = _body_data(env, anchor_cfg, body_cfg)
    count = body_pos.shape[1]
    pos_b, _ = subtract_frame_transforms(
        anchor_pos[:, None].expand(-1, count, -1),
        anchor_quat[:, None].expand(-1, count, -1),
        body_pos,
        body_quat,
    )
    return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg):
    _, anchor_pos, anchor_quat, body_pos, body_quat = _body_data(env, anchor_cfg, body_cfg)
    count = body_pos.shape[1]
    _, ori_b = subtract_frame_transforms(
        anchor_pos[:, None].expand(-1, count, -1),
        anchor_quat[:, None].expand(-1, count, -1),
        body_pos,
        body_quat,
    )
    return matrix_from_quat(ori_b)[..., :2].reshape(env.num_envs, -1)


def robot_body_lin_vel_b(env: ManagerBasedEnv, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg):
    del anchor_cfg
    asset = env.scene[body_cfg.name]
    vel_w = asset.data.body_lin_vel_w.torch[:, body_cfg.body_ids]
    quat_w = asset.data.body_link_quat_w.torch[:, body_cfg.body_ids]
    return quat_apply_inverse(quat_w.reshape(-1, 4), vel_w.reshape(-1, 3)).reshape(env.num_envs, -1)


def robot_body_ang_vel_b(env: ManagerBasedEnv, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg):
    del anchor_cfg
    asset = env.scene[body_cfg.name]
    vel_w = asset.data.body_link_ang_vel_w.torch[:, body_cfg.body_ids]
    quat_w = asset.data.body_link_quat_w.torch[:, body_cfg.body_ids]
    return quat_apply_inverse(quat_w.reshape(-1, 4), vel_w.reshape(-1, 3)).reshape(env.num_envs, -1)
