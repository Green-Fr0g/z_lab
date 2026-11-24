"""Robot link and joint data shared by the kinematics solvers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Hinge:
    """Revolute joint axis and anchor in the child link, with limits in radians."""

    name: str
    axis: torch.Tensor
    anchor: torch.Tensor
    limits: tuple[float, float]


@dataclass(frozen=True)
class Link:
    """One edge's parent-relative pose at zero joint angle, with xyzw orientation."""

    parent: str
    position: torch.Tensor
    orientation: torch.Tensor
    hinge: Hinge | None


@dataclass(frozen=True)
class RobotModel:
    """Floating-root link tree with an explicit policy joint and tracked-link order."""

    root: str
    links: dict[str, Link]
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]

    def path(self, body: str, ancestors: set[str]) -> tuple[str, list[Link]]:
        """Find the nearest ancestor in the given set and return the path from it."""
        path = []
        visited = set()
        while body not in ancestors:
            if body in visited or body not in self.links:
                raise ValueError(f"Link {body!r} does not lead to an expected ancestor")
            visited.add(body)
            link = self.links[body]
            path.append(link)
            body = link.parent
        return body, path[::-1]
