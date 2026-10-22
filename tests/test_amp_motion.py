"""CPU checks for AMP motion loading and motion-based resets."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab.managers import SceneEntityCfg

from z_lab.tasks.amp.mdp.events import init_motion_loader, reset_from_motion_data

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def _write_clip(path, value, fps=50):
    """Recorded order differs from simulation order; the root is not the first recorded body."""
    np.savez(
        path,
        fps=[fps],
        robot_name=["g1"],
        body_names=np.array(["hand", "root"]),
        dof_names=np.array(["elbow", "hip"]),
        root_link_name=["root"],
        body_positions=np.array([[[99.0, 0.0, 0.0], [value, 0.0, 0.8]]], dtype=np.float32),
        body_rotations=np.array([[[0.0, 0.0, 0.0, 1.0]] * 2], dtype=np.float32),
        body_linear_velocities=np.array([[[9.0, 0.0, 0.0], [value + 1, 0.0, 0.0]]], dtype=np.float32),
        body_angular_velocities=np.array([[[8.0, 0.0, 0.0], [0.0, 0.0, value + 2]]], dtype=np.float32),
        dof_positions=np.array([[0.5, value]], dtype=np.float32),
        dof_velocities=np.array([[5.0, value + 3]], dtype=np.float32),
    )


class _Scene(dict):
    env_origins = torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]])


def _env(writes):
    asset = SimpleNamespace(
        joint_names=["hip", "elbow"],
        data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-1.5, 1.5]] * 2] * 3))),
        write_root_pose_to_sim=lambda value, env_ids: writes.update(pose=value, pose_ids=env_ids),
        write_root_velocity_to_sim=lambda value, env_ids: writes.update(velocity=value),
        write_joint_state_to_sim=lambda pos, vel, joint_ids, env_ids: writes.update(pos=pos, vel=vel),
    )
    env = SimpleNamespace(num_envs=3, device="cpu")
    env.scene = _Scene(robot=asset)
    return env


def test_motion_name_alignment_and_reset_write(tmp_path):
    """Body/joint order is resolved by name, so the recorded layout does not leak into the sim."""
    _write_clip(tmp_path / "walk.npz", 1.0)
    writes = {}
    env = _env(writes)
    init_motion_loader(env, None, str(tmp_path), root_name="root", all_body_names=("root", "hand"))

    reset_from_motion_data(
        env,
        torch.tensor([2, 0]),
        str(tmp_path),
        "root",
        ("root", "hand"),
        SceneEntityCfg("robot"),
    )

    assert writes["pose_ids"].tolist() == [2, 0]
    # Root position receives the environment origin; the recorded root is body index 1.
    torch.testing.assert_close(writes["pose"][:, :3], torch.tensor([[31.0, 0.0, 0.8], [11.0, 0.0, 0.8]]))
    torch.testing.assert_close(writes["velocity"], torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 3.0]] * 2))
    # Joints follow the articulation order instead of the recorded order.
    torch.testing.assert_close(writes["pos"], torch.tensor([[1.0, 0.5]] * 2))
    torch.testing.assert_close(writes["vel"], torch.tensor([[4.0, 5.0]] * 2))


def test_clip_boundaries_are_preserved_across_files(tmp_path):
    """Sampling never blends frames of different clips."""
    _write_clip(tmp_path / "a.npz", 1.0)
    _write_clip(tmp_path / "b.npz", 2.0)
    writes = {}
    env = _env(writes)
    init_motion_loader(env, None, str(tmp_path), root_name="root", all_body_names=("root", "hand"))

    # The full reset arrives as a slice; a direct call may also omit the indices.
    reset_from_motion_data(env, slice(None), str(tmp_path), "root", ("root", "hand"), SceneEntityCfg("robot"))
    local_pos = writes["pose"][:, 0] - _Scene.env_origins[:, 0]
    assert torch.isin(local_pos, torch.tensor([1.0, 2.0])).all()
    torch.testing.assert_close(writes["velocity"][:, 0], local_pos + 1)

    writes.clear()
    reset_from_motion_data(env, None, str(tmp_path), "root", ("root", "hand"), SceneEntityCfg("robot"))
    assert len(writes["pose"]) == 3


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"body_names": np.array(["toe", "root"])}, "were not found"),
        ({"body_names": np.array(["hand"])}, "body_names"),
        ({"body_positions": None}, "body_positions"),
    ],
)
def test_schema_errors_are_reported(tmp_path, change, reason):
    path = tmp_path / "broken.npz"
    _write_clip(path, 1.0)
    with np.load(path) as data:
        fields = {key: data[key] for key in data.files if key not in change}
    fields.update({key: value for key, value in change.items() if value is not None})
    np.savez(path, **fields)

    with pytest.raises((ValueError, KeyError), match=reason):
        init_motion_loader(_env({}), None, str(tmp_path), root_name="root", all_body_names=("root", "hand"))
