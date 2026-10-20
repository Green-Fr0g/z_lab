# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-frame raw AMP state in full articulation order, matching the NPZ writer.

Feature selection, coordinate transforms, scaling, and episode history belong to AmpDataset.
The order of these six observation terms is the packed raw state contract with RSL-RL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "robot_body_pos_w",
    "robot_body_quat_w",
    "robot_body_lin_vel_w",
    "robot_body_ang_vel_w",
    "robot_joint_pos",
    "robot_joint_vel",
]


def robot_body_pos_w(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """All body world positions, flattened in simulation body order."""
    return env.scene[asset_name].data.body_pos_w.torch.flatten(1)


def robot_body_quat_w(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """All body world orientations (xyzw), flattened in simulation body order."""
    return env.scene[asset_name].data.body_quat_w.torch.flatten(1)


def robot_body_lin_vel_w(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """All body world linear velocities, using the same accessor as the NPZ exporter."""
    return env.scene[asset_name].data.body_lin_vel_w.torch.flatten(1)


def robot_body_ang_vel_w(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """All body world angular velocities, flattened in simulation body order."""
    return env.scene[asset_name].data.body_ang_vel_w.torch.flatten(1)


def robot_joint_pos(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """All absolute joint positions in simulation joint order."""
    return env.scene[asset_name].data.joint_pos.torch


def robot_joint_vel(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """All absolute joint velocities in simulation joint order."""
    return env.scene[asset_name].data.joint_vel.torch
