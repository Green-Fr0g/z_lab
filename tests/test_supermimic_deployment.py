from types import SimpleNamespace

import onnx
import torch
from onnx.reference import ReferenceEvaluator
from supermimic.models import MultiEncoderAE, TaskSpec
from supermimic.runners.exporter import export_actor
from supermimic.tasks import VelocityTaskCfg, build_actor_obs_td, velocity_goal_from_motion


def test_velocity_goal_uses_centered_body_frame_motion_velocity():
    motion = SimpleNamespace(
        time_step_total=4,
        root_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        root_ang_vel_w=torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4]]),
    )
    angle = torch.tensor(torch.pi / 4)
    root_quat = torch.tensor([[0.0, 0.0, angle.sin(), angle.cos()]])
    cfg = VelocityTaskCfg(
        lin_vel_x_range=(-3.0, 3.0),
        lin_vel_y_range=(-3.0, 3.0),
        ang_vel_z_range=(-1.0, 1.0),
        window_frames=3,
    )

    goal = velocity_goal_from_motion(
        motion,
        torch.tensor([1]),
        root_quat,
        cfg,
        add_noise=False,
    )

    torch.testing.assert_close(goal, torch.tensor([[0.0, -2.0, 0.2]]), atol=1e-6, rtol=0.0)


def test_exporter_writes_motion_and_velocity_policies(tmp_path):
    motion_command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=["pelvis", "torso_link"], anchor_body_name="pelvis"),
    )
    action_term = SimpleNamespace(
        _joint_names=["left_hip", "right_hip"],
        _scale=torch.ones(2),
        _offset=torch.zeros(2),
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: motion_command),
            action_manager=SimpleNamespace(get_term=lambda name: action_term),
            observation_manager=SimpleNamespace(
                active_terms={
                    "actor": ["projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"],
                    "goal": ["command", "motion_anchor_pos_b", "motion_anchor_ori_b"],
                },
                group_obs_term_dim={"actor": [(3,), (3,), (2,), (2,), (2,)], "goal": [(4,), (3,), (6,)]},
            ),
            cfg=SimpleNamespace(
                observations=SimpleNamespace(
                    goal=SimpleNamespace(command=SimpleNamespace(params={"num_frames": 1, "frame_stride": 5}))
                )
            ),
        )
    )
    actor = MultiEncoderAE(
        obs_dim=12,
        action_dim=2,
        task_specs=[TaskSpec("motion", 10), TaskSpec("velocity", 3)],
        slot_mix={"motion": 0.5, "velocity": 0.5},
        latent_dim=4,
        hidden_dims=(8,),
        history_length=2,
    )

    paths = export_actor(actor, env, tmp_path, "run", VelocityTaskCfg())

    assert {path.name for path in paths} == {
        "policy_motion.onnx",
        "policy_motion.pt",
        "policy_velocity.onnx",
        "policy_velocity.pt",
    }

    history = torch.arange(24, dtype=torch.float32).reshape(1, 2, 12) / 10
    packed = torch.cat([part.flatten(1) for part in history.split((3, 3, 2, 2, 2), dim=-1)], dim=-1)
    goals = {"motion": torch.linspace(-1.0, 1.0, 10).unsqueeze(0), "velocity": torch.tensor([[0.1, -0.2, 0.3]])}
    actor.eval()
    for task, goal in goals.items():
        model = onnx.load(str(tmp_path / f"policy_{task}.onnx"))
        onnx.checker.check_model(model)
        assert [value.name for value in model.graph.input] == ["obs"]
        assert [value.name for value in model.graph.output] == ["actions"]
        metadata = {entry.key: entry.value for entry in model.metadata_props}
        assert metadata["task"] == task
        if task == "motion":
            assert metadata["observation_names"].split(",")[-2:] == ["command", "motion_anchor_ori_b"]
            assert metadata["num_goal_frames"] == "1"
        else:
            assert metadata["observation_names"] == "projected_gravity,base_ang_vel,joint_pos,joint_vel,actions,command"
            assert metadata["observation_history_lengths"] == "2,2,2,2,2,1"
            assert metadata["command_names"] == "lin_vel_x,lin_vel_y,ang_vel_z"

        observation = torch.cat((packed, goal), dim=-1)
        actor.set_slot_mix({task: 1.0})
        scripted = torch.jit.load(str(tmp_path / f"policy_{task}.pt"))
        with torch.no_grad():
            expected = actor(build_actor_obs_td({task: goal}, history))
            torch.testing.assert_close(scripted(observation), expected)
        (actions,) = ReferenceEvaluator(model).run(None, {"obs": observation.numpy()})
        torch.testing.assert_close(torch.from_numpy(actions), expected)
