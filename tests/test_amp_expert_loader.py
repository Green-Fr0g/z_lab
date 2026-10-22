"""Expert transitions sampled by the AMP discriminator loader."""

import numpy as np
import pytest
import torch

from z_lab.tasks.amp.utils.motion_loader import AMPBodyStateMotionLoader

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

_IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


def _write_clip(path, root_pos, hand_pos, hand_lin_vel, hand_ang_vel):
    """Write a clip whose recorded body order is ``[hand, root]``, i.e. not the policy order."""
    num_frames = len(root_pos)
    np.savez(
        path,
        fps=[50],
        robot_name=["g1"],
        body_names=np.array(["hand", "root"]),
        dof_names=np.array(["elbow"]),
        root_link_name=["root"],
        body_positions=np.stack([hand_pos, root_pos], axis=1).astype(np.float32),
        body_rotations=np.tile(_IDENTITY_QUAT, (num_frames, 2, 1)).astype(np.float32),
        body_linear_velocities=np.stack([hand_lin_vel, np.zeros((num_frames, 3))], axis=1).astype(np.float32),
        body_angular_velocities=np.stack([hand_ang_vel, np.zeros((num_frames, 3))], axis=1).astype(np.float32),
        dof_positions=np.zeros((num_frames, 1), dtype=np.float32),
        dof_velocities=np.zeros((num_frames, 1), dtype=np.float32),
    )


def _loader(motion_files):
    return AMPBodyStateMotionLoader(
        motion_files,
        body_names=["hand"],
        anchor_name="root",
        all_body_names=["root", "hand"],
        device="cpu",
    )


def test_feature_layout_matches_the_amp_state_observation(tmp_path):
    """Anchor-relative position, rotation columns, and body-frame velocities follow the env layout."""
    _write_clip(
        tmp_path / "walk.npz",
        root_pos=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        hand_pos=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        hand_lin_vel=[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        hand_ang_vel=[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]],
    )

    loader = _loader(tmp_path)
    assert loader.state_dim == 15
    # A two-frame clip has exactly one transition, so every sampled row is that pair.
    states, next_states = next(loader.mini_batch_generator(1, 2))
    # Identity rotation keeps the hand offset; the two rotation columns add [1, 0, 0, 1, 0, 0].
    torch.testing.assert_close(
        states,
        torch.tensor([[1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]] * 2),
    )
    # Consecutive frames become one transition, and the loader never wraps around the clip.
    torch.testing.assert_close(
        next_states,
        torch.tensor([[4.0, 5.0, 5.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]] * 2),
    )


def test_mini_batches_cycle_clips_and_keep_the_transition_pairing(tmp_path):
    """Each mini-batch comes from a single clip and rows stay inside that clip's transitions."""
    _write_clip(
        tmp_path / "a.npz",
        root_pos=np.zeros((3, 3)),
        hand_pos=np.zeros((3, 3)),
        hand_lin_vel=np.full((3, 3), 1.0),
        hand_ang_vel=np.zeros((3, 3)),
    )
    _write_clip(
        tmp_path / "b.npz",
        root_pos=np.zeros((2, 3)),
        hand_pos=np.zeros((2, 3)),
        hand_lin_vel=np.full((2, 3), 9.0),
        hand_ang_vel=np.zeros((2, 3)),
    )

    loader = _loader(tmp_path)
    batches = list(loader.mini_batch_generator(2, 8))
    assert len(batches) == 2
    for (states, next_states), expected_lin_vel in zip(batches, (1.0, 9.0), strict=True):
        assert states.shape == (8, loader.state_dim) and next_states.shape == (8, loader.state_dim)
        torch.testing.assert_close(states[:, 9], torch.full((8,), expected_lin_vel))


def test_unknown_body_names_are_rejected(tmp_path):
    """Feature names must exist in the recorded clips so expert and policy layouts cannot silently diverge."""
    _write_clip(
        tmp_path / "walk.npz",
        root_pos=np.zeros((2, 3)),
        hand_pos=np.zeros((2, 3)),
        hand_lin_vel=np.zeros((2, 3)),
        hand_ang_vel=np.zeros((2, 3)),
    )
    with pytest.raises(ValueError, match="missing bodies"):
        AMPBodyStateMotionLoader(
            str(tmp_path),
            body_names=["toe"],
            anchor_name="root",
            all_body_names=["root", "hand", "toe"],
            device="cpu",
        )
