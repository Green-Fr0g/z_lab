# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared AMP feature registry and xyzw transforms, independent of simulator APIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

RawState = Mapping[str, torch.Tensor]
FeatureFunction = Callable[[RawState, "AmpFeatureEncoder"], torch.Tensor]
AMP_FEATURES: dict[str, FeatureFunction] = {}
DEFAULT_OBS_STRING = ("body_pos_b", "body_ori_b", "body_lin_vel_b", "body_ang_vel_b")


def register_amp_feature(name: str) -> Callable[[FeatureFunction], FeatureFunction]:
    """Register a pure ``(raw_state, encoder) -> (..., feature_width)`` transform."""

    def register(function: FeatureFunction) -> FeatureFunction:
        if not name or name in AMP_FEATURES:
            raise ValueError(f"AMP feature name is empty or already registered: {name!r}.")
        AMP_FEATURES[name] = function
        return function

    return register


def _rotate_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors into a unit xyzw quaternion's local coordinates."""
    xyz, vector = torch.broadcast_tensors(quat[..., :3], vector)
    cross = torch.cross(xyz, vector, dim=-1)
    return vector - 2 * quat[..., 3:] * cross + 2 * torch.cross(xyz, cross, dim=-1)


class AmpFeatureEncoder:
    """Resolve ordered body/joint selections once and apply the same functions to both sources.

    All raw body/joint axes must already follow ``policy_body_names`` / ``policy_joint_names``.
    Features preserve leading sample/history axes. Quaternions are normalized before transforms;
    no default joint offsets or scaling are applied unless the corresponding joint feature is used.
    """

    def __init__(
        self,
        policy_body_names: Sequence[str],
        policy_joint_names: Sequence[str],
        body_names: Sequence[str],
        joint_names: Sequence[str],
        anchor_name: str,
        obs_string: Sequence[str] = DEFAULT_OBS_STRING,
        default_joint_pos: torch.Tensor | None = None,
        default_joint_vel: torch.Tensor | None = None,
        joint_pos_scale: float = 1.0,
        joint_vel_scale: float = 0.05,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        for label, names in (
            ("policy_body_names", policy_body_names),
            ("policy_joint_names", policy_joint_names),
            ("body_names", body_names),
            ("joint_names", joint_names),
            ("obs_string", obs_string),
        ):
            if len(names) != len(set(names)) or (not names and label != "joint_names"):
                raise ValueError(f"{label} must contain unique names and must not be empty.")
        unknown = set(obs_string) - AMP_FEATURES.keys()
        if unknown:
            raise ValueError(f"Unknown AMP features: {sorted(unknown)}.")
        try:
            self.body_ids = torch.tensor([policy_body_names.index(n) for n in body_names], device=self.device)
            self.joint_ids = torch.tensor(
                [policy_joint_names.index(n) for n in joint_names], dtype=torch.long, device=self.device
            )
            self.anchor_id = policy_body_names.index(anchor_name)
        except ValueError as error:
            raise ValueError(f"AMP body/joint/anchor name is missing from the policy metadata: {error}") from error
        self.obs_string = tuple(obs_string)
        self.functions = tuple(AMP_FEATURES[name] for name in self.obs_string)
        self.joint_pos_scale = joint_pos_scale
        self.joint_vel_scale = joint_vel_scale
        self.default_joint_pos = self._joint_default(default_joint_pos, len(policy_joint_names), "joint_pos")
        self.default_joint_vel = self._joint_default(default_joint_vel, len(policy_joint_names), "joint_vel")
        # Infer widths through the registered contract, including user-provided transforms.
        probe = {
            "body_pos_w": torch.zeros(1, len(policy_body_names), 3, device=self.device),
            "body_quat_w": torch.zeros(1, len(policy_body_names), 4, device=self.device),
            "body_lin_vel_w": torch.zeros(1, len(policy_body_names), 3, device=self.device),
            "body_ang_vel_w": torch.zeros(1, len(policy_body_names), 3, device=self.device),
            "dof_pos": torch.zeros(1, len(policy_joint_names), device=self.device),
            "dof_vel": torch.zeros(1, len(policy_joint_names), device=self.device),
        }
        probe["body_quat_w"][..., 3] = 1
        self.frame_dim = self.encode(probe).shape[-1]
        if self.frame_dim < 1:
            raise ValueError("AMP features must produce a nonempty frame.")

    def _joint_default(self, value: torch.Tensor | None, count: int, feature: str) -> torch.Tensor:
        if feature in self.obs_string and (value is None or not self.joint_ids.numel()):
            raise ValueError(f"{feature} requires selected joints and the robot's default joint state.")
        if value is None:
            return torch.zeros(self.joint_ids.numel(), device=self.device)
        value = torch.as_tensor(value, device=self.device, dtype=torch.float32).detach()
        if value.shape != (count,) or not torch.isfinite(value).all():
            raise ValueError(f"Default state for {feature} must be finite with shape ({count},).")
        return value[self.joint_ids]

    def encode(self, raw: RawState) -> torch.Tensor:
        """Concatenate registered features in ``obs_string`` order, preserving leading axes."""
        raw = {**raw, "body_quat_w": F.normalize(raw["body_quat_w"], dim=-1)}
        return torch.cat([function(raw, self) for function in self.functions], dim=-1)


@register_amp_feature("body_pos_b")
def body_pos_b(raw: RawState, encoder: AmpFeatureEncoder) -> torch.Tensor:
    """Body positions relative to the current frame's anchor, in its full local frame."""
    relative = raw["body_pos_w"][..., encoder.body_ids, :] - raw["body_pos_w"][..., encoder.anchor_id, None, :]
    return _rotate_inverse(raw["body_quat_w"][..., encoder.anchor_id, None, :], relative).flatten(-2)


@register_amp_feature("body_ori_b")
def body_ori_b(raw: RawState, encoder: AmpFeatureEncoder) -> torch.Tensor:
    """Anchor-relative rotation matrices, first two columns laid out row by row (6D).

    Matches AMP_mjlab's ``matrix_from_quat(quat_b)[..., :, :2].reshape(..., 6)``, i.e.
    ``(R00, R01, R10, R11, R20, R21)``.
    """
    anchor = raw["body_quat_w"][..., encoder.anchor_id, None, :]
    body = raw["body_quat_w"][..., encoder.body_ids, :]
    # conj(anchor) * body, both xyzw.
    xyz = anchor[..., 3:] * body[..., :3] - body[..., 3:] * anchor[..., :3]
    xyz = xyz - torch.cross(anchor[..., :3].expand_as(body[..., :3]), body[..., :3], dim=-1)
    w = anchor[..., 3:] * body[..., 3:] + (anchor[..., :3] * body[..., :3]).sum(-1, keepdim=True)
    x, y, z = xyz.unbind(-1)
    w = w.squeeze(-1)
    columns = torch.stack(
        (
            1 - 2 * (y * y + z * z),  # R00
            2 * (x * y - z * w),  # R01
            2 * (x * y + z * w),  # R10
            1 - 2 * (x * x + z * z),  # R11
            2 * (x * z - y * w),  # R20
            2 * (y * z + x * w),  # R21
        ),
        dim=-1,
    )
    return columns.flatten(-2)


@register_amp_feature("body_lin_vel_b")
def body_lin_vel_b(raw: RawState, encoder: AmpFeatureEncoder) -> torch.Tensor:
    """Absolute world linear velocities expressed in each body's own local frame."""
    return _rotate_inverse(
        raw["body_quat_w"][..., encoder.body_ids, :], raw["body_lin_vel_w"][..., encoder.body_ids, :]
    ).flatten(-2)


@register_amp_feature("body_ang_vel_b")
def body_ang_vel_b(raw: RawState, encoder: AmpFeatureEncoder) -> torch.Tensor:
    """Absolute world angular velocities expressed in each body's own local frame."""
    return _rotate_inverse(
        raw["body_quat_w"][..., encoder.body_ids, :], raw["body_ang_vel_w"][..., encoder.body_ids, :]
    ).flatten(-2)


@register_amp_feature("body_gravity_b")
def body_gravity_b(raw: RawState, encoder: AmpFeatureEncoder) -> torch.Tensor:
    """Unit downward gravity expressed in the anchor's local frame (3D per frame)."""
    quat = raw["body_quat_w"][..., encoder.anchor_id, :]
    gravity = torch.zeros_like(quat[..., :3])
    gravity[..., 2] = -1
    return _rotate_inverse(quat, gravity)


@register_amp_feature("joint_pos")
def joint_pos(raw: RawState, encoder: AmpFeatureEncoder) -> torch.Tensor:
    """Joint positions relative to the same robot default on both sides."""
    return (raw["dof_pos"][..., encoder.joint_ids] - encoder.default_joint_pos) * encoder.joint_pos_scale


@register_amp_feature("joint_vel")
def joint_vel(raw: RawState, encoder: AmpFeatureEncoder) -> torch.Tensor:
    """Joint velocities relative to the same robot default on both sides."""
    return (raw["dof_vel"][..., encoder.joint_ids] - encoder.default_joint_vel) * encoder.joint_vel_scale
