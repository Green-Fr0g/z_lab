"""Forward kinematics: world-frame pose and velocity of a floating-base robot's
bodies from its root and joint state, differentiable, in the convention of mjlab's
``body_link_pose_w`` and ``body_link_vel_w``.

Each link applies its parent-relative zero pose and then rotates about its
hinge axis through the hinge anchor. Every link is one step.
Each returned body composes the steps on its path from the root, root first, and
sums their hinge twists, taken at the root origin; the root pose and twist apply
last. The pass runs on per-component ``(..., R)`` tensors, which a compiler fuses
into few kernels.
"""

# Adapted from mjlab, Copyright 2025-2026 The MjLab Developers, Apache-2.0.

from __future__ import annotations

import torch

from supermimic.utils.math import quat_apply, quat_mul

from .robot_model import RobotModel

Vec = tuple[torch.Tensor, ...]


class ForwardKinematics:
    """Evaluate the link paths from a floating root in task order."""

    def __init__(
        self,
        model: RobotModel,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        col_of = {name: column for column, name in enumerate(model.joint_names)}
        paths = [model.path(body, {model.root})[1] for body in model.body_names]
        depth = max(map(len, paths))
        # Pad shorter paths with identity steps that read the extra zero joint column.
        constants = torch.zeros((depth, len(paths), 17), dtype=torch.float64)
        constants[..., 12] = 1.0
        columns = torch.full((depth, len(paths)), len(col_of), dtype=torch.long)
        zero = constants.new_zeros(3)
        for row, path in enumerate(paths):
            for step, link in enumerate(path, start=depth - len(path)):
                axis, anchor = zero, zero
                if link.hinge is not None:
                    axis, anchor = link.hinge.axis, link.hinge.anchor
                    columns[step, row] = col_of[link.hinge.name]
                rotation = link.orientation
                constants[step, row] = torch.cat(
                    (
                        link.position + quat_apply(rotation, anchor),
                        quat_apply(rotation, axis),
                        anchor,
                        rotation,
                        quat_mul(rotation, torch.nn.functional.pad(axis, (0, 1))),
                    )
                )
        self.consts = constants.to(device=device, dtype=dtype)
        self.col = columns.to(device=device)
        self.num_bodies = len(paths)

    def __call__(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Positions ``(..., R, 3)``, xyzw quaternions ``(..., R, 4)``, linear and
        angular velocities ``(..., R, 3)`` of the ``R`` bodies, in the world frame,
        from the root link's world pose and velocity and the joints ``(..., J)``."""
        joint_pos, joint_vel = (torch.nn.functional.pad(x, (0, 1)) for x in (joint_pos, joint_vel))
        zero = root_pos.new_zeros((*root_pos.shape[:-1], self.num_bodies))
        one = torch.ones_like(zero)
        pos: Vec = (zero, zero, zero)
        quat: Vec = (zero, zero, zero, one)
        lin: Vec = (zero, zero, zero)
        ang: Vec = (zero, zero, zero)
        for col, consts in zip(self.col, self.consts, strict=True):
            c = consts.unbind(-1)
            half = joint_pos[..., col] / 2
            # The hinge anchor and twist in the root frame, from the parent's pose.
            anchor = _add(pos, _rotate(quat, c[:3]))
            twist = tuple(a * joint_vel[..., col] for a in _rotate(quat, c[3:6]))
            turn = tuple(half.cos() * o + half.sin() * t for o, t in zip(c[9:13], c[13:], strict=True))
            quat = _mul(quat, turn)
            pos = _sub(anchor, _rotate(quat, c[6:9]))
            ang = _add(ang, twist)
            lin = _add(lin, _cross(anchor, twist))

        x, y, z, w = root_quat.unsqueeze(-2).unbind(-1)
        norm = (w * w + x * x + y * y + z * z).sqrt()
        root = (x / norm, y / norm, z / norm, w / norm)
        pos = _rotate(root, pos)
        ang = _add(root_ang_vel.unsqueeze(-2).unbind(-1), _rotate(root, ang))
        lin = _add(root_lin_vel.unsqueeze(-2).unbind(-1), _rotate(root, lin))
        lin = _add(lin, _cross(ang, pos))
        pos = _add(root_pos.unsqueeze(-2).unbind(-1), pos)
        state = torch.stack([*pos, *_mul(root, quat), *lin, *ang], dim=-1)
        pos_w, quat_w, lin_vel_w, ang_vel_w = state.split((3, 4, 3, 3), dim=-1)
        return pos_w, quat_w, lin_vel_w, ang_vel_w


def _add(a: Vec, b: Vec) -> Vec:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _sub(a: Vec, b: Vec) -> Vec:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rotate(q: Vec, v: Vec) -> Vec:
    """``v`` turned by the unit xyzw quaternion ``q``."""
    t = tuple(2 * x for x in _cross(q[:3], v))
    return _add(_add(v, tuple(q[3] * x for x in t)), _cross(q[:3], t))


def _mul(a: Vec, b: Vec) -> Vec:
    """The Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )
