"""Differentiable analytic inverse kinematics for tracked hinge chains."""

from __future__ import annotations

import math

import torch

from supermimic.utils.math import quat_to_matrix, small_matmul

from .robot_model import Hinge, RobotModel

_TWO_PI = 2.0 * math.pi
_SIN_CLAMP = 1e-6


def _atan2(y: torch.Tensor, x: torch.Tensor, damp: float) -> torch.Tensor:
    xd, yd = x.detach(), y.detach()
    return torch.atan2(yd, xd) + (xd * y - yd * x) / (xd.square() + yd.square() + damp)


def _damped(value: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:
    return surrogate + (value - surrogate).detach()


def _reduce_chain(
    model: RobotModel,
    body: str,
    row_of: dict[str, int],
    col_of: dict[str, int],
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    ancestor, path = model.path(body, set(row_of) - {body})
    fixed_rotation = torch.eye(3, dtype=torch.float64)
    axes: list[torch.Tensor] = []
    joints: list[Hinge] = []
    for link in path:
        fixed_rotation = fixed_rotation @ quat_to_matrix(link.orientation)
        if link.hinge is not None:
            axes.append(fixed_rotation @ link.hinge.axis)
            joints.append(link.hinge)
    count = len(joints)
    if count not in (1, 2, 3):
        raise ValueError(f"tracked body {body!r} has {count} hinges between tracked links")

    first_axis = axes[0]
    if count == 1:
        basis = torch.eye(3, dtype=first_axis.dtype)
        perpendicular = torch.linalg.cross(first_axis, basis[first_axis.abs().argmin()])
        second_axis = perpendicular / perpendicular.norm()
    else:
        second_axis = axes[1]
        if abs(float(first_axis @ second_axis)) > 1e-5:
            raise ValueError(f"tracked body {body!r} needs orthogonal first and second hinge axes")
    third_axis = torch.linalg.cross(first_axis, second_axis)
    frame = torch.stack((first_axis, second_axis, third_axis))
    sign = torch.zeros(3, dtype=torch.float64)
    sign[:count] = 1.0
    if count == 3:
        alignment = float(axes[2] @ third_axis)
        if abs(abs(alignment) - 1.0) > 1e-5:
            raise ValueError(f"tracked body {body!r} needs three orthogonal hinge axes")
        sign[2] = math.copysign(1.0, alignment)
    lower = torch.zeros(3, dtype=torch.float64)
    upper = torch.zeros(3, dtype=torch.float64)
    lower[:count], upper[:count] = torch.tensor([joint.limits for joint in joints], dtype=torch.float64).T
    # Center fully bounded joints on their range, and unbounded joints on zero.
    center = torch.where(torch.isfinite(lower) & torch.isfinite(upper), (lower + upper) / 2, 0.0)
    columns = [col_of[joint.name] for joint in joints] + [-1] * (3 - count)
    return (
        row_of[ancestor],
        frame.T,
        (frame @ fixed_rotation).T,
        sign,
        center,
        lower,
        upper,
        columns,
    )


class AnalyticIK:
    """Recover joint angles and rates from tracked-body rotations and angular velocities.

    Each tracked child and its nearest tracked ancestor bound a chain of one to
    three orthogonal hinges. The precomputed frames turn each chain's relative
    rotation into an XYZ Euler product.
    """

    def __init__(
        self,
        model: RobotModel,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        tracked = model.body_names
        if len(tracked) < 2:
            raise ValueError("analytic IK requires at least one tracked link beyond the floating root")
        row_of = {body: row for row, body in enumerate(tracked)}
        col_of = {name: col for col, name in enumerate(model.joint_names)}
        parent, a, b, sign, center, lower, upper, columns = zip(
            *(_reduce_chain(model, body, row_of, col_of) for body in tracked[1:]), strict=True
        )
        slot_columns = torch.tensor(columns).flatten()
        used_columns = slot_columns[slot_columns >= 0]
        if used_columns.sort().values.tolist() != list(range(len(col_of))):
            raise ValueError("tracked body chains must cover every selected joint exactly once")
        self.parent = torch.tensor(parent, device=device)
        self.child = torch.arange(1, len(tracked), device=device)
        self.A, self.B, self.sign, self.angle_center, self.lo, self.hi = (
            torch.stack(values).to(device=device, dtype=dtype) for values in (a, b, sign, center, lower, upper)
        )
        self.three = self.sign[:, 2] != 0
        self.one = self.sign[:, 1] == 0
        self.perm = slot_columns.argsort()[slot_columns.lt(0).sum() :].to(device)

    def __call__(
        self,
        rotations: torch.Tensor,
        angular_velocities: torch.Tensor,
        damp: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positions [rad] and velocities [rad/s] in the configured joint order."""
        parent_rotation = small_matmul(rotations[..., self.parent, :, :], self.A).transpose(-1, -2)
        child_rotation = small_matmul(rotations[..., self.child, :, :], self.B)
        relative_rotation = small_matmul(parent_rotation, child_rotation)
        relative_velocity = angular_velocities[..., self.child, :] - angular_velocities[..., self.parent, :]
        velocity = small_matmul(parent_rotation, relative_velocity.unsqueeze(-1)).squeeze(-1)

        u1_x = _atan2(
            relative_rotation[..., 2, 1] - relative_rotation[..., 1, 2],
            relative_rotation[..., 1, 1] + relative_rotation[..., 2, 2],
            4 * damp,
        )
        u1_xy = _atan2(relative_rotation[..., 2, 1], relative_rotation[..., 1, 1], damp)
        u2_xy = _atan2(relative_rotation[..., 0, 2], relative_rotation[..., 0, 0], damp)
        u1 = _atan2(-relative_rotation[..., 1, 2], relative_rotation[..., 2, 2], damp)
        u2 = torch.asin(relative_rotation[..., 0, 2].clamp(-1 + _SIN_CLAMP, 1 - _SIN_CLAMP))
        u3 = _atan2(-relative_rotation[..., 0, 1], relative_rotation[..., 0, 0], damp)
        angles = torch.where(
            self.three[..., None],
            torch.stack((u1, u2, u3), dim=-1),
            torch.stack((torch.where(self.one, u1_x, u1_xy), u2_xy, torch.zeros_like(u3)), dim=-1),
        )
        alternate = torch.stack((u1 + math.pi, math.pi - u2, u3 + math.pi), dim=-1)
        q = self._qpos(angles)
        alternate_q = self._qpos(alternate)
        use_alternate = (self.three & (self._cost(alternate_q) < self._cost(q)))[..., None]
        angles = torch.where(use_alternate, alternate, angles)

        numerator = relative_rotation[..., 1, 2] * velocity[..., 1] + relative_rotation[..., 2, 2] * velocity[..., 2]
        cosine_squared = relative_rotation[..., 1, 2].square() + relative_rotation[..., 2, 2].square()
        u3d = _damped(numerator / cosine_squared.clamp(min=damp), numerator / (cosine_squared + damp))
        u3d = torch.where(self.three, u3d, 0.0)
        u2d = torch.cos(angles[..., 0]) * velocity[..., 1] + torch.sin(angles[..., 0]) * velocity[..., 2]
        u1d = velocity[..., 0] - relative_rotation[..., 0, 2] * u3d
        rates = self.sign * torch.stack((u1d, u2d, u3d), dim=-1)
        return self._qpos(angles).flatten(-2)[..., self.perm], rates.flatten(-2)[..., self.perm]

    def _qpos(self, angles: torch.Tensor) -> torch.Tensor:
        qpos = self.sign * angles
        return qpos - _TWO_PI * torch.round((qpos - self.angle_center) / _TWO_PI)

    def _cost(self, qpos: torch.Tensor) -> torch.Tensor:
        violation = (self.lo - qpos).clamp(min=0) + (qpos - self.hi).clamp(min=0)
        return (violation + 1e-3 * qpos.abs()).sum(-1)
