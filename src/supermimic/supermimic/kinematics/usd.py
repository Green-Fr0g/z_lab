"""Read kinematic data from an OpenUSD robot prim."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from supermimic.utils.math import quat_apply, quat_inv, quat_mul

from .robot_model import Hinge, Link, RobotModel

if TYPE_CHECKING:
    from pxr import Gf, Usd, UsdPhysics


def load_robot_model(
    robot_prim: Usd.Prim,
    joint_names: Sequence[str],
    body_names: Sequence[str],
) -> RobotModel:
    """Read the floating-root tree, with the root link first in ``body_names``.

    Use the default prim for a standalone asset or the robot prim in a composed
    scene. Only the subset connecting the root to the tracked links is needed.
    """
    from pxr import UsdGeom, UsdPhysics

    if not robot_prim.IsValid():
        raise ValueError("Expected a valid USD robot prim")
    if not body_names or len(set(body_names)) != len(body_names):
        raise ValueError("Tracked body names must be unique and start at the root link")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("Policy joint names must be unique")

    stage = robot_prim.GetStage()
    # OpenUSD 26.08 renamed this API. Isaac Sim can supply an earlier OpenUSD.
    parse = getattr(UsdPhysics, "UsdPhysicsLoadStageFromPrimRange", None)
    if parse is None:
        parse = UsdPhysics.LoadUsdPhysicsFromRange
    descriptors = parse(stage, [robot_prim.GetPath()])
    kinds = UsdPhysics.ObjectType
    body_paths = set(descriptors.get(kinds.RigidBody, ((), ()))[0])
    body_paths_by_name = {path.name: path for path in body_paths}
    if len(body_paths_by_name) != len(body_paths):
        raise ValueError(f"Rigid body names must be unique under {robot_prim.GetPath()}")
    missing_bodies = set(body_names) - body_paths_by_name.keys()
    if missing_bodies:
        raise ValueError(f"Tracked links missing from {robot_prim.GetPath()}: {sorted(missing_bodies)}")

    joint_by_child = {}
    for _, items in descriptors.values():
        for joint in items:
            if not isinstance(joint, UsdPhysics.JointDesc) or not joint.jointEnabled or joint.excludeFromArticulation:
                continue
            if joint.body1 not in body_paths:
                continue
            if joint.body1 in joint_by_child:
                raise ValueError(f"Multiple incoming joints for USD link {joint.body1}")
            joint_by_child[joint.body1] = joint

    root = body_names[0]
    if body_paths_by_name[root] in joint_by_child:
        raise ValueError(f"First tracked link {root!r} must be the floating root")
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    links = {}
    hinges = set()
    for body in body_names[1:]:
        visited = set()
        while body != root and body not in links:
            visited.add(body)
            joint = joint_by_child.get(body_paths_by_name[body])
            if joint is None or joint.body0 not in body_paths or joint.body0.name in visited:
                raise ValueError(f"Tracked link {body!r} has no tree path to floating root {root!r}")
            if not joint.isValid:
                raise ValueError(f"Invalid USD joint {joint.primPath}")
            link = _read_link(joint, meters_per_unit)
            if link.hinge is not None:
                if link.hinge.name in hinges:
                    raise ValueError(f"Duplicate joint name {link.hinge.name!r}")
                hinges.add(link.hinge.name)
            links[body] = link
            body = link.parent

    missing_joints = hinges - set(joint_names)
    if missing_joints:
        raise ValueError(f"Tracked link paths require joints missing from the policy order: {sorted(missing_joints)}")
    return RobotModel(root, links, tuple(joint_names), tuple(body_names))


def _read_link(joint: UsdPhysics.JointDesc, meters_per_unit: float) -> Link:
    from pxr import UsdPhysics

    def quaternion(value: Gf.Quatf) -> torch.Tensor:
        components = (*value.GetImaginary(), value.GetReal())
        return torch.nn.functional.normalize(torch.tensor(components, dtype=torch.float64), dim=-1)

    parent_position = torch.tensor(tuple(joint.localPose0Position), dtype=torch.float64) * meters_per_unit
    child_position = torch.tensor(tuple(joint.localPose1Position), dtype=torch.float64) * meters_per_unit
    child_orientation = quaternion(joint.localPose1Orientation)
    orientation = quat_mul(quaternion(joint.localPose0Orientation), quat_inv(child_orientation))
    position = parent_position - quat_apply(orientation, child_position)

    hinge = None
    if isinstance(joint, UsdPhysics.RevoluteJointDesc):
        axis = torch.zeros(3, dtype=torch.float64)
        axis[(UsdPhysics.Axis.X, UsdPhysics.Axis.Y, UsdPhysics.Axis.Z).index(joint.axis)] = 1.0
        limits = (-math.inf, math.inf)
        if joint.limit.enabled:
            limits = math.radians(joint.limit.lower), math.radians(joint.limit.upper)
        hinge = Hinge(joint.primPath.name, quat_apply(child_orientation, axis), child_position, limits)
    elif not isinstance(joint, UsdPhysics.FixedJointDesc):
        raise ValueError(f"Expected a fixed or revolute USD joint at {joint.primPath}")
    return Link(joint.body0.name, position, orientation, hinge)
