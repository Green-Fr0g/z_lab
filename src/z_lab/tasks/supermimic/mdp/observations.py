"""State and reference-window observations for SuperMimic tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import joint_pos_rel
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def biased_joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return nominal-relative positions as measured by the encoders."""
    positions = joint_pos_rel(env, asset_cfg)
    bias = getattr(env, "supermimic_encoder_bias", None)
    return positions if bias is None else positions + bias[:, asset_cfg.joint_ids]


def gencoord_state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return root pose, joint positions, root twist, and joint velocities."""
    data = env.scene[asset_cfg.name].data
    return torch.cat(
        (
            data.root_link_pos_w.torch - env.scene.env_origins,
            data.root_link_quat_w.torch,
            data.joint_pos.torch[:, asset_cfg.joint_ids],
            data.root_link_lin_vel_w.torch,
            data.root_link_ang_vel_w.torch,
            data.joint_vel.torch[:, asset_cfg.joint_ids],
        ),
        dim=-1,
    )


def maxcoord_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Return flattened blocks of link positions, orientations, and twists."""
    command: MotionCommand = env.command_manager.get_term("motion")
    data = command.robot.data
    body_ids = command.body_indexes
    pos = data.body_link_pos_w.torch[:, body_ids] - env.scene.env_origins[:, None, :]
    quat = data.body_link_quat_w.torch[:, body_ids]
    lin_vel = data.body_link_lin_vel_w.torch[:, body_ids]
    ang_vel = data.body_link_ang_vel_w.torch[:, body_ids]
    return torch.cat((pos.flatten(1), quat.flatten(1), lin_vel.flatten(1), ang_vel.flatten(1)), dim=-1)


def motion_joint_window(env: ManagerBasedEnv, num_frames: int, frame_stride: int) -> torch.Tensor:
    """Return the reference joint position and velocity at each lookahead frame."""
    command: MotionCommand = env.command_manager.get_term("motion")
    frames = _goal_frames(command, num_frames, frame_stride)
    joint_pos = command.motion.joint_pos[frames[..., None], command.joint_indexes]
    joint_vel = command.motion.joint_vel[frames[..., None], command.joint_indexes]
    return torch.cat((joint_pos, joint_vel), dim=-1).flatten(1)


def motion_anchor_pos_window(env: ManagerBasedEnv, num_frames: int, frame_stride: int) -> torch.Tensor:
    """Return reference pelvis positions in the current pelvis frame."""
    command: MotionCommand = env.command_manager.get_term("motion")
    frames = _goal_frames(command, num_frames, frame_stride)
    target_pos = command.motion.root_pos_w[frames]
    root_pos = command.robot_anchor_pos_w - env.scene.env_origins
    root_quat = command.robot_anchor_quat_w[:, None, :].expand(-1, num_frames, -1)
    return quat_apply_inverse(root_quat, target_pos - root_pos[:, None, :]).flatten(1)


def motion_anchor_ori_window(env: ManagerBasedEnv, num_frames: int, frame_stride: int) -> torch.Tensor:
    """Return reference pelvis orientations in the current pelvis frame as 6D rotations."""
    command: MotionCommand = env.command_manager.get_term("motion")
    frames = _goal_frames(command, num_frames, frame_stride)
    target_quat = command.motion.root_quat_w[frames]
    root_quat_inv = quat_inv(command.robot_anchor_quat_w)[:, None, :].expand_as(target_quat)
    relative_quat = quat_mul(root_quat_inv, target_quat)
    return matrix_from_quat(relative_quat)[..., :2].flatten(1)


def _goal_frames(command: MotionCommand, num_frames: int, frame_stride: int) -> torch.Tensor:
    offsets = torch.arange(num_frames, device=command.time_steps.device) * frame_stride
    last = command.motion.clip_end(command.time_steps) - 1
    return torch.minimum(command.time_steps[:, None] + offsets, last[:, None])
