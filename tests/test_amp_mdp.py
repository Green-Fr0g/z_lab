"""Regression tests for AMP recovery rewards, contact filtering, and velocity curricula."""

from types import SimpleNamespace

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.string import resolve_matching_names

from z_lab.tasks.amp.config.g1_23.amp_env_cfg import UnitreeG123AmpEnvCfg


def test_rewards_switch_only_during_delayed_failure():
    """Healthy recovery-enabled robots retain normal rewards until a failure is pending."""
    cfg = UnitreeG123AmpEnvCfg()
    data = SimpleNamespace(
        body_link_quat_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * 4)),
        body_link_pos_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.8]]] * 4)),
        body_link_lin_vel_w=SimpleNamespace(torch=torch.zeros(4, 1, 3)),
        body_link_ang_vel_w=SimpleNamespace(torch=torch.zeros(4, 1, 3)),
        default_root_state=torch.tensor([[0.0, 0.0, 0.8]] * 4),
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: torch.zeros(4, 3)),
        termination_manager=SimpleNamespace(
            _delay_env_mask=torch.tensor([False, True, True, False]),
            _delay_counters=torch.tensor([0, 1, 0, 1]),
        ),
    )
    for name in ("track_anchor_linear_velocity", "track_anchor_angular_velocity", "body_ang_vel_xy_l2"):
        term = getattr(cfg.rewards, name)
        for value in term.params.values():
            if isinstance(value, SceneEntityCfg):
                value.body_ids = [0]
        torch.testing.assert_close(term.func(env, **term.params), torch.tensor([1.0, 0.0, 1.0, 1.0]))

    height_term = cfg.rewards.track_root_height
    height_term.params["asset_cfg"].body_ids = [0]
    torch.testing.assert_close(height_term.func(env, **height_term.params), torch.tensor([0.0, 3.5, 0.0, 0.0]))
    env.termination_manager = SimpleNamespace()
    torch.testing.assert_close(height_term.func(env, **height_term.params), torch.zeros(4))


def test_undesired_contacts_excludes_ankles():
    """Foot support is allowed while knee contact still incurs the configured penalty."""
    cfg = UnitreeG123AmpEnvCfg()
    term = cfg.rewards.undesired_contacts
    sensor_cfg = term.params["sensor_cfg"]
    body_names = [
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "left_knee_link",
        "torso_link",
    ]
    sensor_cfg.body_ids, _ = resolve_matching_names(sensor_cfg.body_names, body_names)
    forces = torch.zeros(2, 4, len(body_names), 3)
    forces[:, :, :4, 2] = 100.0
    forces[1, 0, 4, 2] = 11.0
    data = SimpleNamespace(net_normal_forces_w_history=SimpleNamespace(torch=forces))
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": SimpleNamespace(data=data)}))
    reward = term.weight * term.func(env, **term.params)
    torch.testing.assert_close(reward, torch.tensor([0.0, -0.1]))


def test_velocity_curriculum_reaches_limits_without_compounding():
    """Expand both endpoints by training progress, regardless of reset frequency."""
    cfg = UnitreeG123AmpEnvCfg()
    term = SimpleNamespace(cfg=cfg.commands.base_velocity)
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: term),
        common_step_counter=0,
        device="cpu",
    )
    curriculum = cfg.curriculum.command_vel
    update = curriculum.func(curriculum, env)
    cases = (
        (0, ((-0.5, 1.0), (-0.5, 0.5), (-1.0, 1.0))),
        (30_000, ((-0.75, 1.5), (-0.625, 0.625), (-1.1425, 1.1425))),
        (60_000, ((-1.0, 2.0), (-0.75, 0.75), (-1.285, 1.285))),
        (60_000, ((-1.0, 2.0), (-0.75, 0.75), (-1.285, 1.285))),
        (120_000, ((-1.5, 3.0), (-1.0, 1.0), (-1.57, 1.57))),
        (240_000, ((-1.5, 3.0), (-1.0, 1.0), (-1.57, 1.57))),
    )
    for step, expected in cases:
        env.common_step_counter = step
        update(env, [0], **curriculum.params)
        actual = (term.cfg.ranges.lin_vel_x, term.cfg.ranges.lin_vel_y, term.cfg.ranges.ang_vel_z)
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))

    # A separately configured task must use its own endpoints and expansion duration.
    custom_cfg = UnitreeG123AmpEnvCfg()
    term.cfg = custom_cfg.commands.base_velocity
    term.cfg.limit_ranges.lin_vel_x = (-2.0, 4.0)
    custom_cfg.curriculum.command_vel.params["num_steps"] = 100
    curriculum = custom_cfg.curriculum.command_vel
    update = curriculum.func(curriculum, env)
    env.common_step_counter = 50
    update(env, [0], **curriculum.params)
    assert term.cfg.ranges.lin_vel_x == (-1.25, 2.5)
