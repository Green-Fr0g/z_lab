"""Observable contracts for the SuperMimic algorithm foundation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pxr import Usd
from supermimic.algorithms.learners.policy_learner import ActorCriticPolicyLearner, ActorCriticPolicyLearnerCfg
from supermimic.algorithms.learners.world_model_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.algorithms.models import Dynamics, MultiEncoderAE, TaskSpec, build_mlp
from supermimic.algorithms.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.algorithms.representations import GenCoordRepresentation, MaxCoordRepresentation, StateRepresentation
from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.kinematics import ForwardKinematics, load_robot_model
from tensordict import TensorDict

from z_lab.assets.robots.g1 import G1_DOF_ORDER, G1_MIMIC_LINK_NAMES, G1_USD_PATH
from z_lab.tasks.supermimic.mdp.commands import MotionLibrary


@pytest.fixture
def g1_fk():
    stage = Usd.Stage.Open(str(G1_USD_PATH))
    model = load_robot_model(stage.GetDefaultPrim(), G1_DOF_ORDER, G1_MIMIC_LINK_NAMES)
    return ForwardKinematics(model)


def test_gencoord_representation_round_trips_full_state():
    num_joints = 29
    representation = GenCoordRepresentation(
        num_joints=num_joints,
        default_joint_pos=torch.zeros(num_joints),
        dt=0.02,
        decimation=4,
    )
    assert isinstance(representation, StateRepresentation)
    root_quat = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    full_state = torch.randn(8, 13 + 2 * num_joints)
    full_state[:, 3:7] = root_quat

    rep_state = representation.represent(full_state)
    reconstructed = representation.reconstruct_full_state(representation.extract_aux_state(full_state), rep_state)

    assert rep_state.shape == (8, 10 + 2 * num_joints)
    torch.testing.assert_close(reconstructed, full_state, atol=1e-6, rtol=1e-6)

    conflicting_aux = representation.extract_aux_state(full_state).clone()
    conflicting_aux[:, 2] += 1.0
    reconstructed_from_rep_height = representation.reconstruct_full_state(conflicting_aux, rep_state)
    torch.testing.assert_close(reconstructed_from_rep_height[:, 2], rep_state[:, 0])

    wrong_joint_count_state = torch.zeros(2, 13 + 2 * (num_joints - 1))
    with pytest.raises(ValueError, match="full_state must end in"):
        representation.represent(wrong_joint_count_state)


def test_maxcoord_representation_round_trips_link_states():
    num_bodies = 5
    representation = MaxCoordRepresentation(
        num_bodies=num_bodies,
        default_joint_pos=torch.zeros(8),
        dt=0.02,
        decimation=4,
    )
    assert isinstance(representation, StateRepresentation)
    body_positions = torch.randn(3, num_bodies, 3)
    body_quaternions = torch.nn.functional.normalize(torch.randn(3, num_bodies, 4), dim=-1)
    body_linear_velocities = torch.randn(3, num_bodies, 3)
    body_angular_velocities = torch.randn(3, num_bodies, 3)
    full_state = torch.cat(
        (
            body_positions.flatten(1),
            body_quaternions.flatten(1),
            body_linear_velocities.flatten(1),
            body_angular_velocities.flatten(1),
        ),
        dim=-1,
    )

    representation_state = representation.represent(full_state)
    reconstructed = representation.reconstruct_full_state(
        representation.extract_aux_state(full_state), representation_state
    )

    torch.testing.assert_close(
        representation.represent(reconstructed),
        representation_state,
        atol=2e-6,
        rtol=2e-6,
    )
    assert representation_state.shape == (3, num_bodies * 15 + 4)


def test_maxcoord_prediction_integrates_root_velocity_delta():
    representation = MaxCoordRepresentation(
        num_bodies=1,
        default_joint_pos=torch.zeros(1),
        dt=0.1,
        decimation=1,
    )
    full_state = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    aux_state = representation.extract_aux_state(full_state)
    rep_state = representation.represent(full_state)
    prediction = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    next_aux, next_rep = representation.apply_prediction(aux_state, rep_state, prediction)

    torch.testing.assert_close(next_aux, torch.tensor([[0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]))
    assert next_rep.shape == (1, 19)


def test_replay_samples_only_episode_contiguous_transition_windows():
    replay = ReplayBuffer(
        {
            "state": FieldSpec(1),
            "terminated": FieldSpec(dtype=torch.bool),
            "truncated": FieldSpec(dtype=torch.bool),
        },
        buffer_size=4,
        num_envs=1,
    )
    for step in range(8):
        replay.insert(
            TensorDict(
                {
                    "state": torch.tensor([[step]], dtype=torch.float32),
                    "terminated": torch.tensor([step in (3, 7)]),
                    "truncated": torch.tensor([False]),
                },
                batch_size=[1],
            )
        )

    windows = replay.sample(sequence_length=3, batch_size=128)

    assert windows["state"].shape == (128, 3, 1)
    assert not windows["terminated"][:, :-1].any()
    assert not windows["truncated"][:, :-1].any()
    assert torch.all(windows["state"][:, 1:, 0] == windows["state"][:, :-1, 0] + 1)


def test_g1_forward_kinematics_matches_packed_motion(g1_fk):
    motion_path = Path("src/z_lab/tasks/mimic/config/g1/motions/dance.npz")
    frames = np.asarray([0, 173, 521, 1042, 1456])

    with np.load(motion_path) as motion:
        joint_indices = [list(motion["dof_names"]).index(name) for name in G1_DOF_ORDER]
        body_indices = [list(motion["body_names"]).index(name) for name in G1_MIMIC_LINK_NAMES]
        root_index = body_indices[0]
        root_pos = torch.from_numpy(motion["body_positions"][frames, root_index])
        root_quat = torch.from_numpy(motion["body_rotations"][frames, root_index])
        joint_pos = torch.from_numpy(motion["dof_positions"][frames][:, joint_indices])
        joint_vel = torch.from_numpy(motion["dof_velocities"][frames][:, joint_indices])
        root_lin_vel = torch.from_numpy(motion["body_linear_velocities"][frames, root_index])
        root_ang_vel = torch.from_numpy(motion["body_angular_velocities"][frames, root_index])
        expected_pos = torch.from_numpy(motion["body_positions"][frames][:, body_indices])
        expected_quat = torch.from_numpy(motion["body_rotations"][frames][:, body_indices])

    actual_pos, actual_quat, _, _ = g1_fk(
        root_pos,
        root_quat,
        joint_pos,
        root_lin_vel,
        root_ang_vel,
        joint_vel,
    )

    torch.testing.assert_close(actual_pos, expected_pos, atol=3e-6, rtol=0.0)
    orientation_agreement = (actual_quat * expected_quat).sum(dim=-1).abs()
    assert torch.all(orientation_agreement > 0.99999)


def test_world_model_learner_updates_dynamics_from_replay_sequence():
    num_joints = 3
    batch_size = 8
    sequence_length = 4
    representation = GenCoordRepresentation(
        num_joints=num_joints,
        default_joint_pos=torch.zeros(num_joints),
        dt=0.02,
        decimation=4,
    )
    dynamics = Dynamics(
        representation,
        num_actions=num_joints,
        sim_context_dim=2,
        action_scale=torch.ones(num_joints),
        action_offset=torch.zeros(num_joints),
        hidden_dims=(32,),
    )
    learner = WorldModelLearner(
        dynamics,
        representation,
        WorldModelLearnerCfg(learning_rate=1e-3),
    )
    states = torch.randn(batch_size, sequence_length, representation.full_state_dim)
    states[..., 6] = 1.0
    sequence = TensorDict(
        {
            "state": states,
            "action": torch.rand(batch_size, sequence_length, num_joints),
            "sim_ctx": torch.randn(batch_size, sequence_length, 2),
        },
        batch_size=[batch_size, sequence_length],
    )
    initial_weights = [parameter.detach().clone() for parameter in dynamics.parameters()]

    losses = learner.update([sequence])

    assert torch.isfinite(losses["Loss/state"])
    assert any(
        not torch.equal(before, after) for before, after in zip(initial_weights, dynamics.parameters(), strict=True)
    )


def test_actor_critic_updates_through_g1_model_rollout(g1_fk):
    motion_file = Path("src/z_lab/tasks/mimic/config/g1/motions/dance.npz")
    motion = MotionLibrary(
        [str(motion_file)],
        G1_MIMIC_LINK_NAMES,
        "pelvis",
        step_dt=0.02,
        device="cpu",
    )
    num_joints = len(G1_DOF_ORDER)
    representation = GenCoordRepresentation(
        num_joints=num_joints,
        default_joint_pos=torch.zeros(num_joints),
        dt=0.02,
        decimation=4,
    )
    representation.forward_kinematics = g1_fk
    dynamics = Dynamics(
        representation,
        num_actions=num_joints,
        sim_context_dim=1,
        action_scale=torch.ones(num_joints),
        action_offset=torch.zeros(num_joints),
        hidden_dims=(32,),
    )
    frame = 160
    root_pos = motion.root_pos_w[frame]
    root_quat = motion.root_quat_w[frame]
    full_state = torch.cat(
        (
            root_pos,
            root_quat,
            motion.joint_pos[frame],
            motion.root_lin_vel_w[frame],
            motion.root_ang_vel_w[frame],
            motion.joint_vel[frame],
        )
    )
    replay = ReplayBuffer(
        {
            "state": FieldSpec(representation.full_state_dim),
            "action": FieldSpec(num_joints),
            "prev_action": FieldSpec(num_joints),
            "motion_time": FieldSpec(dtype=torch.long),
            "terminated": FieldSpec(dtype=torch.bool),
            "truncated": FieldSpec(dtype=torch.bool),
            "sim_ctx": FieldSpec(1),
        },
        buffer_size=8,
        num_envs=1,
    )
    replay.insert(
        TensorDict(
            {
                "state": full_state.unsqueeze(0),
                "action": torch.zeros(1, num_joints),
                "prev_action": torch.zeros(1, num_joints),
                "motion_time": torch.tensor([frame]),
                "terminated": torch.tensor([False]),
                "truncated": torch.tensor([False]),
                "sim_ctx": torch.zeros(1, 1),
            },
            batch_size=[1],
        )
    )
    joint_limits = torch.tensor([[-3.0, 3.0]]).expand(num_joints, -1)
    reward_weights = {
        "motion_global_anchor_pos": 0.5,
        "motion_global_anchor_ori": 0.5,
        "motion_body_pos": 1.0,
        "motion_body_ori": 1.0,
        "motion_body_lin_vel": 1.0,
        "motion_body_ang_vel": 1.0,
        "action_rate_l2": -0.1,
        "joint_limit": -2.0,
    }
    reward_stds = {
        "motion_global_anchor_pos": 0.3,
        "motion_global_anchor_ori": 0.4,
        "motion_body_pos": 0.3,
        "motion_body_ori": 0.4,
        "motion_body_lin_vel": 1.0,
        "motion_body_ang_vel": 3.14,
    }
    task_specs = [TaskSpec("motion", 640), TaskSpec("velocity", 3)]
    world_env = MultiTaskWorldModelEnv(
        dynamics,
        representation,
        SimpleNamespace(motion=motion, joint_indexes=torch.arange(num_joints)),
        replay,
        batch_size=2,
        soft_joint_pos_limits=joint_limits,
        reward_weights=reward_weights,
        reward_stds=reward_stds,
        anchor_pos_threshold=0.25,
        anchor_ori_threshold=0.8,
        task_specs=task_specs,
    )
    actor = MultiEncoderAE(
        obs_dim=representation.proprio_obs_dim(num_joints),
        action_dim=num_joints,
        task_specs=task_specs,
        slot_mix={"motion": 0.5, "velocity": 0.5},
        latent_dim=8,
        hidden_dims=(32,),
    )
    critic = build_mlp(representation.rep_state_dim + 670 + 1, (16,), 1)
    learner = ActorCriticPolicyLearner(
        actor,
        critic,
        world_env,
        ActorCriticPolicyLearnerCfg(
            batch_size=2,
            horizon=2,
            critic_updates=1,
            critic_mini_batches=2,
        ),
    )
    actor_weights = [parameter.detach().clone() for parameter in actor.parameters()]
    velocity_encoder_weights = [parameter.detach().clone() for parameter in actor.encoders["velocity"].parameters()]

    metrics = learner.update()

    assert torch.isfinite(metrics["Policy/reward"])
    assert torch.isfinite(metrics["Policy/actor_grad_norm"])
    assert any(not torch.equal(before, after) for before, after in zip(actor_weights, actor.parameters(), strict=True))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(velocity_encoder_weights, actor.encoders["velocity"].parameters(), strict=True)
    )
