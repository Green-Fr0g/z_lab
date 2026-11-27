"""AMP feature projection for expert, simulator, and model states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..representations import StateRepresentation


class AMPFeatures:
    """Reconstruct the discriminator's ordered body features from model state."""

    def __init__(
        self,
        representation: StateRepresentation,
        anchor_index: int,
        body_indices: torch.Tensor,
    ) -> None:
        self.representation = representation
        self.anchor_index = anchor_index
        self.body_indices = body_indices
        self.observation_dim = 15 * body_indices.numel()

    def from_full_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode recorded states, including terminal simulator frames."""
        return self.from_bodies(
            *self.representation.world_frame_bodies(
                self.representation.extract_aux_state(state), self.representation.represent(state)
            )
        )

    def from_bodies(
        self, positions: torch.Tensor, rotations: torch.Tensor, linear: torch.Tensor, angular: torch.Tensor
    ) -> torch.Tensor:
        """Encode decoded link poses and link-origin velocities."""
        ids = self.body_indices
        body_rot = rotations.index_select(-3, ids)
        body_lin = linear.index_select(-2, ids)
        body_ang = angular.index_select(-2, ids)
        return body_amp_features(
            positions[..., self.anchor_index, :],
            rotations[..., self.anchor_index, :, :],
            positions.index_select(-2, ids),
            body_rot,
            body_lin,
            body_ang,
        )


def body_amp_features(
    anchor_pos_w: torch.Tensor,
    anchor_rot_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_rot_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
) -> torch.Tensor:
    """Encode ordered body poses in the anchor frame and twists in each body's frame.

    Rotations are world-frame rotation matrices. Linear velocities are measured
    at the link origins.
    """
    world_to_anchor = anchor_rot_w.transpose(-1, -2).unsqueeze(-3)
    relative_pos = (world_to_anchor @ (body_pos_w - anchor_pos_w.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
    relative_rot = (world_to_anchor @ body_rot_w)[..., :2]
    world_to_body = body_rot_w.transpose(-1, -2)
    linear = (world_to_body @ body_lin_vel_w.unsqueeze(-1)).squeeze(-1)
    angular = (world_to_body @ body_ang_vel_w.unsqueeze(-1)).squeeze(-1)
    return torch.cat(
        (relative_pos.flatten(-2), relative_rot.flatten(-3), linear.flatten(-2), angular.flatten(-2)), dim=-1
    )
