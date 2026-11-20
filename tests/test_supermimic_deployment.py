from types import SimpleNamespace

import onnx
import torch
from onnx.reference import ReferenceEvaluator
from supermimic.algorithms.models import MultiEncoderAE, TaskSpec
from supermimic.runners.exporter import export_actor_as_onnx
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


def test_exporter_writes_motion_and_velocity_onnx_contracts(tmp_path):
    class Motion:
        time_step_total = 4
        joint_pos = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        joint_vel = torch.ones_like(joint_pos)

        def body_state(self, frames):
            count = len(frames)
            pos = torch.zeros(count, 2, 3)
            quat_xyzw = torch.zeros(count, 2, 4)
            quat_xyzw[..., 3] = 1.0
            velocity = torch.zeros_like(pos)
            return pos, quat_xyzw, velocity, velocity

    motion = Motion()
    motion_command = SimpleNamespace(
        motion=motion,
        joint_indexes=torch.tensor([1, 0]),
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
                observations=SimpleNamespace(goal=SimpleNamespace(command=SimpleNamespace(params={"frame_stride": 5})))
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

    paths = export_actor_as_onnx(actor, env, tmp_path, "run", VelocityTaskCfg())

    assert [path.name for path in paths] == ["policy_motion.onnx", "policy_velocity.onnx"]
    motion_model, velocity_model = (onnx.load(str(path)) for path in paths)
    onnx.checker.check_model(motion_model)
    onnx.checker.check_model(velocity_model)
    assert [value.name for value in motion_model.graph.input] == ["obs", "time_step"]
    assert [value.name for value in motion_model.graph.output] == [
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]
    assert [value.name for value in velocity_model.graph.input] == ["obs"]
    assert [value.name for value in velocity_model.graph.output] == ["actions"]
    metadata = {entry.key: entry.value for entry in velocity_model.metadata_props}
    assert metadata["observation_names"] == "projected_gravity,base_ang_vel,joint_pos,joint_vel,actions,command"
    assert metadata["observation_history_lengths"] == "2,2,2,2,2,1"
    assert metadata["command_names"] == "lin_vel_x,lin_vel_y,ang_vel_z"

    history = torch.arange(24, dtype=torch.float32).reshape(1, 2, 12) / 10
    packed = torch.cat([part.flatten(1) for part in history.split((3, 3, 2, 2, 2), dim=-1)], dim=-1)
    velocity = torch.tensor([[0.1, -0.2, 0.3]])
    actor.eval()
    actor.set_slot_mix({"velocity": 1.0})
    with torch.no_grad():
        expected = actor(build_actor_obs_td({"velocity": velocity}, history))
    velocity_input = torch.cat((packed, velocity), dim=-1).numpy()
    (actions,) = ReferenceEvaluator(velocity_model).run(None, {"obs": velocity_input})
    torch.testing.assert_close(torch.from_numpy(actions), expected)

    motion_input = torch.cat((packed, torch.zeros(1, 10)), dim=-1).numpy()
    outputs = ReferenceEvaluator(motion_model).run(None, {"obs": motion_input, "time_step": torch.ones(1, 1).numpy()})
    torch.testing.assert_close(torch.from_numpy(outputs[1]), torch.tensor([[3.0, 2.0]]))
