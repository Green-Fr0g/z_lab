"""AMP expert loading, shared feature semantics, and episode-aware rollout contracts."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from rsl_rl.utils import AmpDataset, AmpFeatureEncoder

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def _write_motion(path, length, base, **overrides):
    # Recorded body/joint order differs from policy order; every field encodes frame identity.
    t = np.arange(length, dtype=np.float32) + base
    root = np.stack([t, np.zeros(length), np.full(length, 0.8)], axis=-1)
    fields = dict(
        fps=[50],
        body_names=["hand", "root"],
        dof_names=["elbow", "hip"],
        root_link_name=["root"],
        body_positions=np.stack([root + 10, root], axis=1),
        body_rotations=np.tile([0.0, 0.0, 0.0, 1.0], (length, 2, 1)),
        body_linear_velocities=np.stack([root + 50, root + 40], axis=1),
        body_angular_velocities=np.stack([root + 70, root + 60], axis=1),
        dof_positions=np.stack([t + 30, t + 20], axis=1),
        dof_velocities=np.stack([t + 90, t + 80], axis=1),
    )
    fields.update(overrides)
    np.savez(path, **fields)


def _dataset(path, **kwargs):
    return AmpDataset(
        str(path),
        ["root", "hand"],
        ["hip", "elbow"],
        "root",
        **{"num_envs": 2, "num_steps_per_env": 3, "step_dt": 0.02, **kwargs},
    )


def _pack(raw):
    # Public packed observation order, independent of dataset internals.
    return torch.cat(
        [
            raw[name].flatten(1)
            for name in (
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "dof_pos",
                "dof_vel",
            )
        ],
        dim=-1,
    )


def test_expert_loading_alignment_stride_and_recovery_mix(tmp_path):
    _write_motion(tmp_path / "a.npz", 7, 0)
    _write_motion(tmp_path / "b.npz", 11, 100)
    _write_motion(tmp_path / "short.npz", 1, 200)
    _write_motion(tmp_path / "fall.npz", 9, 1000)
    dataset = _dataset(tmp_path, num_frames=3, history_stride=2, recovery_sample_ratio=0.25)
    raw = dataset.sample_raw_frames(32)
    root_x = raw["body_pos_w"][:, :, 0, 0]
    assert raw["body_pos_w"].shape == (32, 3, 2, 3)
    assert int((root_x[:, 0] >= 1000).sum()) == 8
    assert not ((root_x >= 200) & (root_x < 1000)).any()
    torch.testing.assert_close(root_x[:, 1:] - root_x[:, :-1], torch.full((32, 2), 2.0))
    assert (root_x[root_x[:, 0] < 100, -1] <= 6).all()
    assert (root_x[(root_x[:, 0] >= 100) & (root_x[:, 0] < 200), -1] <= 110).all()
    assert (root_x[root_x[:, 0] >= 1000, -1] <= 1008).all()
    torch.testing.assert_close(raw["body_pos_w"][:, :, 1, 0], root_x + 10)
    torch.testing.assert_close(raw["body_lin_vel_w"][:, :, 0, 0], root_x + 40)
    torch.testing.assert_close(raw["body_ang_vel_w"][:, :, 0, 0], root_x + 60)
    torch.testing.assert_close(raw["dof_pos"][:, :, 0], root_x + 20)
    torch.testing.assert_close(raw["dof_vel"][:, :, 0], root_x + 80)
    assert dataset.raw_dim == 30
    assert dataset.obs_dim == 90
    raw["dof_pos"].zero_()
    assert dataset.sample_raw_frames(1)["dof_pos"].min() >= 20
    assert dataset.motion_names == ("a", "b", "short")
    assert dataset.sample_expert(32).shape == (32, 90)
    for ratio in (0.0, 1.0):
        endpoint = _dataset(tmp_path, recovery_sample_ratio=ratio)
        is_recovery = endpoint.sample_raw_frames(4)["body_pos_w"][:, 0, 0, 0] >= 1000
        assert is_recovery.tolist() == [bool(ratio)] * 4


def test_expert_files_uniform_despite_lengths_and_empty_recovery(tmp_path):
    _write_motion(tmp_path / "short.npz", 5, 0)
    _write_motion(tmp_path / "long.npz", 101, 100)
    dataset = _dataset(tmp_path, num_frames=3, history_stride=2, recovery_sample_ratio=1.0)
    assert not dataset.uses_recovery
    with torch.random.fork_rng():
        torch.manual_seed(42)
        root_x = dataset.sample_raw_frames(8192)["body_pos_w"][:, :, 0, 0]
    assert 0.46 < (root_x[:, 0] < 100).float().mean().item() < 0.54
    torch.testing.assert_close(root_x[root_x[:, 0] < 100][0], torch.tensor([0.0, 2.0, 4.0]))
    long_starts = root_x[root_x[:, 0] >= 100, 0]
    assert long_starts.min() == 100
    assert long_starts.max() == 196
    with pytest.raises(ValueError, match="no clip long enough"):
        _dataset(tmp_path, num_frames=52, history_stride=2)


def test_expert_and_policy_share_nontrivial_coordinate_and_name_semantics(tmp_path):
    # Anchor yaw=90 degrees; hand relative roll=90 degrees. Use nonunit raw quaternions too.
    s = 2**-0.5
    raw = {
        "body_pos_w": torch.tensor([[[10.0, 20.0, 2.0], [8.0, 21.0, 5.0]]]),
        "body_quat_w": torch.tensor([[[0.0, 0.0, 2 * s, 2 * s], [1.0, 1.0, 1.0, 1.0]]]),
        "body_lin_vel_w": torch.tensor([[[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]]),
        "body_ang_vel_w": torch.tensor([[[2.0, 3.0, 4.0], [6.0, 4.0, 5.0]]]),
        "dof_pos": torch.tensor([[1.2, 0.3]]),
        "dof_vel": torch.tensor([[3.0, 4.0]]),
    }
    np.savez(
        tmp_path / "pose.npz",
        fps=[50],
        root_link_name=["root"],
        body_names=["hand", "root"],
        dof_names=["elbow", "hip"],
        **{
            key: raw[name][:, [1, 0]].numpy()
            for name, key in (
                ("body_pos_w", "body_positions"),
                ("body_quat_w", "body_rotations"),
                ("body_lin_vel_w", "body_linear_velocities"),
                ("body_ang_vel_w", "body_angular_velocities"),
                ("dof_pos", "dof_positions"),
                ("dof_vel", "dof_velocities"),
            )
        },
    )
    dataset = _dataset(tmp_path, num_envs=1, num_frames=1, body_names=["hand"])
    expected = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    torch.testing.assert_close(dataset.sample_expert(1), expected, atol=2e-6, rtol=1e-5)
    dataset.initialize_policy(_pack(raw))
    torch.testing.assert_close(
        dataset.add_policy_step(_pack(raw), torch.tensor([False])), expected, atol=2e-6, rtol=1e-5
    )

    # Apply a global yaw=90 degrees and a translation, and reverse the producer's name order.
    rotated = {name: value.clone() for name, value in raw.items()}
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for name in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        rotated[name] = raw[name] @ rotation.T
    rotated["body_pos_w"] += torch.tensor([7.0, -2.0, 1.0])
    rotated["body_quat_w"] = torch.tensor([[[0.0, 0.0, 1.0, 0.0], [0.0, s, s, 0.0]]])
    rotated = {name: value[:, [1, 0]] for name, value in rotated.items()}
    reordered = AmpFeatureEncoder(["hand", "root"], ["elbow", "hip"], ["hand"], ["hip"], "root")
    torch.testing.assert_close(reordered.encode(rotated), expected, atol=2e-6, rtol=1e-5)

    optional = AmpFeatureEncoder(
        ["root", "hand"],
        ["hip", "elbow"],
        ["hand"],
        ["elbow", "hip"],
        "root",
        obs_string=["joint_vel", "body_gravity_b", "joint_pos"],
        default_joint_pos=torch.tensor([1.0, 0.1]),
        default_joint_vel=torch.tensor([1.0, 2.0]),
        joint_pos_scale=2.0,
        joint_vel_scale=0.5,
    )
    torch.testing.assert_close(
        optional.encode(raw), torch.tensor([[1.0, 1.0, 0.0, 0.0, -1.0, 0.4, 0.4]]), atol=2e-6, rtol=1e-5
    )
    # Preserve history axes and project gravity with the anchor, independent of selected bodies.
    history = {name: value[:, None].expand(-1, 2, *value.shape[1:]).clone() for name, value in raw.items()}
    history["body_quat_w"][:, 1, 0] = torch.tensor([2 * s, 0.0, 0.0, 2 * s])
    torch.testing.assert_close(
        optional.encode(history),
        torch.tensor([[[1.0, 1.0, 0.0, 0.0, -1.0, 0.4, 0.4], [1.0, 1.0, 0.0, -1.0, 0.0, 0.4, 0.4]]]),
        atol=2e-6,
        rtol=1e-5,
    )


def test_policy_history_padding_done_stride_and_cross_rollout(tmp_path):
    _write_motion(tmp_path / "a.npz", 10, 0)
    dataset = _dataset(
        tmp_path,
        num_frames=3,
        history_stride=2,
        obs_string=["joint_vel"],
        joint_names=["hip"],
        default_joint_vel=torch.zeros(2),
        joint_vel_scale=1.0,
    )

    def frame(a, b):
        raw = torch.zeros(2, 30)
        raw[:, 6 + 3] = raw[:, 6 + 7] = 1  # Packed xyzw orientations.
        raw[:, -2] = torch.tensor([a, b])
        return raw

    dataset.initialize_policy(frame(0, 100))
    no_done = torch.tensor([False, False])
    for t, history in ((1, [0, 0, 1]), (2, [0, 0, 2]), (3, [0, 1, 3])):
        result = dataset.add_policy_step(frame(t, 100 + t), no_done)
        torch.testing.assert_close(result[0], torch.tensor(history, dtype=torch.float32))
    with pytest.raises(OverflowError):
        dataset.add_policy_step(frame(4, 104), no_done)
    dataset.clear_rollout()
    torch.testing.assert_close(dataset.add_policy_step(frame(4, 104), no_done)[0], torch.tensor([0.0, 2.0, 4.0]))
    with pytest.raises(ValueError, match="final_raw_obs"):
        dataset.add_policy_step(frame(50, 105), torch.tensor([True, False]))
    terminal = dataset.add_policy_step(frame(50, 105), torch.tensor([True, False]), frame(5, 105))
    torch.testing.assert_close(terminal, torch.tensor([[1.0, 3.0, 5.0], [101.0, 103.0, 105.0]]))
    result = dataset.add_policy_step(frame(51, 106), no_done)
    torch.testing.assert_close(result, torch.tensor([[50.0, 50.0, 51.0], [102.0, 104.0, 106.0]]))
    batches = list(dataset.mini_batch_generator(4, 2))
    assert len(batches) == 8
    assert [len(policy) for policy, _ in batches[:4]] == [2, 2, 1, 1]
    assert all(policy.shape == expert.shape for policy, expert in batches)
    # Every policy sample participates once per epoch, including uneven splits.
    expected = sorted([0.0, 100.0, 1.0, 101.0, 50.0, 102.0])
    for epoch in (batches[:4], batches[4:]):
        assert sorted(torch.cat([policy[:, 0] for policy, _ in epoch]).tolist()) == expected


@pytest.mark.parametrize(
    "invalid, message",
    [
        ({"fps": [100]}, "frequency"),
        ({"dof_names": ["elbow", "unknown"]}, "missing dof_names"),
        ({"dof_velocities": np.zeros((3, 2))}, "shape"),
        ({"body_names": ["root", "root"]}, "duplicate"),
    ],
)
def test_expert_npz_schema_errors_identify_clip(tmp_path, invalid, message):
    _write_motion(tmp_path / "broken.npz", 2, 0, **invalid)
    with pytest.raises(ValueError, match=f"broken.npz.*{message}"):
        _dataset(tmp_path)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_frames": True}, "num_frames"),
        ({"history_stride": 1.5}, "history_stride"),
        ({"recovery_sample_ratio": float("nan")}, "recovery_sample_ratio"),
        ({"obs_string": ["unknown"]}, "Unknown AMP"),
        ({"body_names": ["hand", "hand"]}, "unique"),
        ({"obs_string": ["joint_pos"]}, "default joint state"),
    ],
)
def test_invalid_amp_configuration(tmp_path, kwargs, message):
    _write_motion(tmp_path / "a.npz", 2, 0)
    with pytest.raises(ValueError, match=message):
        _dataset(tmp_path, **kwargs)


def test_amp_ppo_updates_dataset_rollouts_and_terminal_rewards(tmp_path):
    """Exercise real PPO/discriminator updates without a simulator, including Same-Step autoreset."""
    from rsl_rl.algorithms import AMPPPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict

    _write_motion(tmp_path / "a.npz", 8, 0)
    dataset = _dataset(
        tmp_path,
        num_frames=2,
        obs_string=["joint_vel"],
        joint_names=["hip"],
        default_joint_vel=torch.zeros(2),
        joint_vel_scale=1.0,
    )
    obs = TensorDict({"policy": torch.zeros(2, 3), "critic": torch.zeros(2, 4)}, batch_size=[2])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(
        obs,
        groups,
        "actor",
        2,
        hidden_dims=[16],
        distribution_cfg={"class_name": "rsl_rl.modules.distribution:GaussianDistribution", "init_std": 0.5},
    )
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[16])
    storage = RolloutStorage("rl", 2, 3, obs, [2])
    algorithm = AMPPPO(
        actor,
        critic,
        storage,
        num_learning_epochs=2,
        num_mini_batches=2,
        amp_cfg={"amp_dataset": dataset, "discriminator_cfg": {"hidden_dims": [16], "task_reward_lerp": 0.5}},
    )

    # Each optimizer step must carry finite gradients and change all three networks together.
    models = (algorithm.actor, algorithm.critic, algorithm.discriminator)
    step_weights = []
    joint_steps = []

    def before_step(optimizer, args, kwargs):
        step_weights.clear()
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        for model in models:
            parameters = list(model.parameters())
            assert all(id(p) in optimized for p in parameters)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            assert any(p.grad.abs().sum() > 0 for p in parameters)
            step_weights.append([p.detach().clone() for p in parameters])

    def after_step(optimizer, args, kwargs):
        for model, before in zip(models, step_weights, strict=True):
            assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
        joint_steps.append(1)

    before_handle = algorithm.optimizer.register_step_pre_hook(before_step)
    after_handle = algorithm.optimizer.register_step_post_hook(after_step)

    def frame(velocities):
        packed = torch.zeros(2, 30)
        packed[:, 9] = packed[:, 13] = 1
        packed[:, -2] = velocities
        return packed

    velocities = torch.tensor([0.0, 10.0])
    obs["amp_policy"] = frame(velocities)
    initial_weights = [p.detach().clone() for p in algorithm.discriminator.parameters()]
    for rollout in range(2):
        for step in range(3):
            with torch.inference_mode():
                algorithm.act(obs)
                terminal_velocity = velocities + 1
                done = torch.tensor([step == 1, False])
                velocities = terminal_velocity.clone()
                if done[0]:
                    velocities[0] = 50 + rollout * 100
                next_obs = obs.clone()
                next_obs["amp_policy"] = frame(velocities)
                final_obs = next_obs.clone()
                final_obs["amp_policy"] = frame(terminal_velocity)
                extras = {"final_obs": final_obs, "time_outs": done}
                # The raw final frame determines the reward even though next_obs contains reset state.
                algorithm.process_env_step(next_obs, torch.ones(2), done, extras)
                torch.testing.assert_close(dataset.policy_history[step, :, -1], terminal_velocity)
                obs = next_obs
        assert dataset.num_samples == 6
        assert storage.step == 3
        assert "amp_policy" not in storage.observations
        with torch.inference_mode():
            algorithm.compute_returns(obs)
        losses = algorithm.update()
        expected_steps = (rollout + 1) * algorithm.num_learning_epochs * algorithm.num_mini_batches
        assert len(joint_steps) == expected_steps
        assert all(group["lr"] == algorithm.learning_rate for group in algorithm.optimizer.param_groups)
        assert all(state["step"] == expected_steps for state in algorithm.optimizer.state.values())
        assert all(np.isfinite(value) for value in losses.values())
        assert losses["amp_grad_pen"] >= 0
        assert dataset.num_samples == storage.step == 0
        assert dataset.policy_initialized
        assert algorithm.discriminator.obs_normalizer.count == 12 * (rollout + 1)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in algorithm.discriminator.parameters())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_weights, algorithm.discriminator.parameters(), strict=True)
    )
    before_handle.remove()
    after_handle.remove()
    # Checkpoints restore both networks and their shared optimizer, including AMP moments.
    saved = deepcopy(algorithm.save())
    with torch.no_grad():
        for model in models:
            for parameter in model.parameters():
                parameter.add_(1)
    algorithm.optimizer.state.clear()
    algorithm.load(saved, None, strict=True)
    for model, key in zip(
        models, ("actor_state_dict", "critic_state_dict", "amp_discriminator_state_dict"), strict=True
    ):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[key][name])
    assert all(state["step"] == expected_steps for state in algorithm.optimizer.state.values())
    algorithm.eval_mode()
    count = algorithm.discriminator.obs_normalizer.count.clone()
    algorithm.discriminator.update_normalization(torch.zeros(2, 2), torch.zeros(2, 2))
    assert algorithm.discriminator.obs_normalizer.count == count
