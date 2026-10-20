"""CPU checks for raw motion loading, reset writes, and delayed termination semantics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab.managers import TerminationManager, TerminationTermCfg

from z_lab.tasks.amp.mdp.events import init_amp_motion, init_delayed_termination, reset_from_amp_motion
from z_lab.tasks.amp.mdp.motion_loader import MotionLoader
from z_lab.tasks.amp.mdp.terminations import DelayedTerminationManager

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def _write_clip(path, value, fps=50):
    # Recorded order differs from simulation order; the root is not the first recorded body.
    np.savez(
        path,
        fps=[fps],
        body_names=["hand", "root"],
        dof_names=["elbow", "hip"],
        root_link_name=["root"],
        body_positions=np.array([[[99.0, 0.0, 0.0], [value, 0.0, 0.8]]], dtype=np.float32),
        body_rotations=np.array([[[0.0, 0.0, 0.0, 1.0]] * 2], dtype=np.float32),
        body_linear_velocities=np.array([[[9.0, 0.0, 0.0], [value + 1, 0.0, 0.0]]], dtype=np.float32),
        body_angular_velocities=np.array([[[8.0, 0.0, 0.0], [0.0, 0.0, value + 2]]], dtype=np.float32),
        dof_positions=np.array([[0.5, value]], dtype=np.float32),
        dof_velocities=np.array([[5.0, value + 3]], dtype=np.float32),
    )


def _load(path):
    return MotionLoader(str(path), body_names=["root", "hand"], joint_names=["hip", "elbow"], root_name="root")


def test_motion_name_alignment_and_recovery_selection(tmp_path):
    """NPZ body/joint order, raw metadata, and reset state selection agree by name."""
    _write_clip(tmp_path / "walk.npz", 1.0)
    _write_clip(tmp_path / "FALL.npz", 2.0, fps=100)
    loader = _load(tmp_path)
    assert loader.normal.motion_names == ("walk",)
    assert loader.normal.counts.tolist() == [1]
    assert loader.normal.offsets.tolist() == [0]
    assert loader.normal.fps.tolist() == [50]
    assert loader.recovery.fps.tolist() == [100]
    torch.testing.assert_close(loader.normal.body_pos_w[0], torch.tensor([[1.0, 0.0, 0.8], [99.0, 0.0, 0.0]]))
    mask = torch.tensor([False, True])
    state = loader.sample_reset_state(2, mask)
    torch.testing.assert_close(state["root_pos"], torch.tensor([[1.0, 0.0, 0.8], [2.0, 0.0, 0.8]]))
    torch.testing.assert_close(state["root_quat"], torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2))
    torch.testing.assert_close(state["root_lin_vel"], torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    torch.testing.assert_close(state["root_ang_vel"], torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]]))
    torch.testing.assert_close(state["dof_pos"], torch.tensor([[1.0, 0.5], [2.0, 0.5]]))
    torch.testing.assert_close(state["dof_vel"], torch.tensor([[4.0, 5.0], [5.0, 5.0]]))
    state["dof_pos"].zero_()
    assert loader.normal.dof_pos[0, 0] == 1.0
    assert mask.tolist() == [False, True]


def test_missing_recovery_falls_back_and_preserves_clip_boundaries(tmp_path):
    _write_clip(tmp_path / "a.npz", 1.0)
    _write_clip(tmp_path / "b.npz", 2.0)
    loader = _load(tmp_path)
    assert loader.normal.motion_names == ("a", "b")
    assert loader.normal.offsets.tolist() == [0, 1]
    state = loader.sample_reset_state(16, torch.ones(16, dtype=torch.bool))
    assert torch.isin(state["root_pos"][:, 0], torch.tensor([1.0, 2.0])).all()
    torch.testing.assert_close(state["dof_vel"][:, 0], state["root_pos"][:, 0] + 3)


@pytest.mark.parametrize("invalid", ["missing_joint", "root_mismatch", "frame_mismatch"])
def test_motion_schema_errors_include_clip_name(tmp_path, invalid):
    path = tmp_path / "broken.npz"
    _write_clip(path, 1.0)
    with np.load(path) as data:
        fields = dict(data)
    if invalid == "missing_joint":
        fields["dof_names"] = np.array(["elbow", "unknown"])
        reason = "missing joints"
    elif invalid == "root_mismatch":
        fields["root_link_name"] = ["hand"]
        reason = "differs from"
    else:
        fields["dof_velocities"] = np.zeros((2, 2), dtype=np.float32)
        reason = "shape"
    np.savez(path, **fields)
    with pytest.raises(ValueError, match=f"broken.npz.*{reason}"):
        _load(tmp_path)


def _termination_env():
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        fall=torch.zeros(3, dtype=torch.bool),
        timeout=torch.zeros(3, dtype=torch.bool),
    )
    env.termination_manager = TerminationManager(
        {
            "fall": TerminationTermCfg(func=lambda env: env.fall),
            "timeout": TerminationTermCfg(func=lambda env: env.timeout, time_out=True),
        },
        env,
    )
    return env


def test_delay_recovery_expiry_timeout_and_partial_reset():
    """Use real termination terms to exercise all outcomes of a recovery window."""
    env = _termination_env()
    manager = DelayedTerminationManager(env.termination_manager, torch.tensor([True, True, False]), 3)
    env.fall[:] = True
    assert manager.compute().tolist() == [False, False, True]
    assert manager.terminated.tolist() == [False, False, True]
    assert manager.get_term("fall").tolist() == [True, True, True]
    env.fall[0] = False
    assert manager.compute().tolist() == [False, False, True]
    assert manager.delay_env_succeeded.tolist() == [True, False, False]
    assert manager.delay_counters.tolist() == [0, 2, 0]
    assert manager.compute().tolist() == [False, True, True]
    manager.reset(torch.tensor([0]))
    assert manager.delay_env_succeeded.tolist() == [False, False, False]
    assert manager.delay_counters.tolist() == [0, 3, 0]
    manager.reset(slice(None))
    env.fall[:] = True
    manager.compute()
    env.timeout[:2] = True
    env.fall[0] = False
    assert manager.compute().tolist() == [True, True, True]
    assert manager.time_outs.tolist() == [True, True, False]
    assert manager.delay_counters.tolist() == [0, 0, 0]
    assert not manager.delay_env_succeeded.any()


def test_motion_startup_and_reset_write_selected_environments(tmp_path):
    """Exercise real sampling/events; only the simulator write boundary is replaced by a recorder."""
    _write_clip(tmp_path / "walk.npz", 1.0)
    _write_clip(tmp_path / "fall.npz", 2.0)
    writes = {}
    asset = SimpleNamespace(
        body_names=["root", "hand"],
        joint_names=["hip", "elbow"],
        data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-1.5, 1.5]] * 2] * 3))),
        write_root_pose_to_sim=lambda value, env_ids: writes.update(pose=value, pose_ids=env_ids),
        write_root_velocity_to_sim=lambda value, env_ids: writes.update(velocity=value),
        write_joint_state_to_sim=lambda pos, vel, env_ids: writes.update(pos=pos, vel=vel),
    )
    env = _termination_env()
    env.scene = {"robot": asset}
    init_amp_motion(env, None, str(tmp_path), root_name="root")

    # Scene supports both named assets and environment origins.
    class Scene(dict):
        env_origins = torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]])

    env.scene = Scene(env.scene)
    init_delayed_termination(env, None, delay_reset_env_ratio=1.0, max_delay_steps=3)
    reset_from_amp_motion(env, [2, 0])
    assert writes["pose_ids"].tolist() == [2, 0]
    torch.testing.assert_close(writes["pose"][:, :3], torch.tensor([[32.0, 0.0, 0.8], [12.0, 0.0, 0.8]]))
    torch.testing.assert_close(writes["pos"], torch.tensor([[1.5, 0.5]] * 2))
    torch.testing.assert_close(writes["velocity"], torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 4.0]] * 2))
    torch.testing.assert_close(writes["vel"], torch.tensor([[5.0, 5.0]] * 2))
    reset_from_amp_motion(env, slice(1, 2), use_recovery=False)
    torch.testing.assert_close(writes["pose"][:, :3], torch.tensor([[21.0, 0.0, 0.8]]))
    writes.clear()
    reset_from_amp_motion(env, [])
    assert not writes
