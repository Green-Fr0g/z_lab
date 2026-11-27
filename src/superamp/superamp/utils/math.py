"""Differentiable quaternion and state helpers using xyzw quaternions."""

from __future__ import annotations

import torch


def small_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply small matrices using elementwise arithmetic without TF32 rounding."""
    return (left.unsqueeze(-1) * right.unsqueeze(-3)).sum(-2)


def quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the Hamilton product of xyzw quaternions."""
    lx, ly, lz, lw = left.unbind(-1)
    rx, ry, rz, rw = right.unbind(-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def quat_inv(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the inverse of unit xyzw quaternions."""
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)


def quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit xyzw quaternions."""
    vector_part = quaternion[..., :3]
    cross = torch.linalg.cross(vector_part, vector, dim=-1)
    return vector + 2.0 * (quaternion[..., 3:] * cross + torch.linalg.cross(vector_part, cross, dim=-1))


def quat_apply_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the inverse of unit xyzw quaternions."""
    return quat_apply(quat_inv(quaternion), vector)


def quat_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized xyzw quaternions to rotation matrices."""
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    x, y, z, w = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def matrix_to_quat(rotation: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to xyzw quaternions without dynamic indexing."""
    m00, m01, m02 = rotation[..., 0, 0], rotation[..., 0, 1], rotation[..., 0, 2]
    m10, m11, m12 = rotation[..., 1, 0], rotation[..., 1, 1], rotation[..., 1, 2]
    m20, m21, m22 = rotation[..., 2, 0], rotation[..., 2, 1], rotation[..., 2, 2]
    magnitude = torch.sqrt(
        torch.stack(
            (
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
                1.0 + m00 + m11 + m22,
            ),
            dim=-1,
        ).clamp_min(1e-10)
    )
    candidates = torch.stack(
        (
            torch.stack((magnitude[..., 0].square(), m10 + m01, m02 + m20, m21 - m12), dim=-1),
            torch.stack((m10 + m01, magnitude[..., 1].square(), m12 + m21, m02 - m20), dim=-1),
            torch.stack((m20 + m02, m21 + m12, magnitude[..., 2].square(), m10 - m01), dim=-1),
            torch.stack((m21 - m12, m02 - m20, m10 - m01, magnitude[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    candidates = candidates / (2.0 * magnitude[..., None].clamp_min(0.1))
    best = magnitude.argmax(dim=-1)[..., None, None].expand(*magnitude.shape[:-1], 1, 4)
    return candidates.gather(-2, best).squeeze(-2)


def quat_to_rot6d(quaternion: torch.Tensor) -> torch.Tensor:
    """Represent a quaternion by the first two columns of its rotation matrix."""
    return quat_to_matrix(quaternion)[..., :2].flatten(-2)


def rot6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Convert the first two flattened rotation columns to a rotation matrix."""
    columns = rotation.unflatten(-1, (3, 2))
    first = torch.nn.functional.normalize(columns[..., 0], dim=-1)
    second = columns[..., 1] - (first * columns[..., 1]).sum(-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def rodrigues_precompute(
    rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose axis-angle vectors into unit axes and sine/cosine values."""
    angle = torch.sqrt(rotation.square().sum(-1, keepdim=True) + 1e-8)
    axis = rotation / angle
    return axis, torch.cos(angle), torch.sin(angle)


def rodrigues_apply(
    vector: torch.Tensor,
    axis: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Rotate vectors using a precomputed axis and sine/cosine values."""
    axis_cross_vector = torch.linalg.cross(axis, vector, dim=-1)
    axis_dot_vector = (axis * vector).sum(-1, keepdim=True)
    return cosine * vector + sine * axis_cross_vector + (1.0 - cosine) * axis_dot_vector * axis


def rotation_error_magnitude(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return angular distance in radians for pairs of rotation matrices."""
    trace = (left * right).sum(dim=(-2, -1))
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cosine)


def quat_yaw(quaternion: torch.Tensor) -> torch.Tensor:
    """Extract the world-Z yaw component of xyzw quaternions."""
    x, y, z, w = quaternion.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    return torch.stack(
        (torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw * 0.5), torch.cos(yaw * 0.5)), dim=-1
    )


def quat_integrate(quaternion: torch.Tensor, angular_velocity: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate an xyzw orientation by a world-frame angular velocity."""
    rotation = angular_velocity * dt
    angle = torch.sqrt(rotation.square().sum(dim=-1, keepdim=True) + 1e-8)
    half_angle = angle * 0.5
    axis = rotation / angle
    delta = torch.cat((torch.sin(half_angle) * axis, torch.cos(half_angle)), dim=-1)
    return torch.nn.functional.normalize(quat_mul(delta, quaternion), dim=-1)
