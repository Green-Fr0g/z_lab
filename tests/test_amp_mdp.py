"""Regression tests for AMP recovery rewards, contact filtering, and velocity curricula."""

from types import SimpleNamespace
from unittest.mock import Mock

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


def test_recovery_assistance_delays_start_and_decay_without_restarting_on_reset():
    """Delay assistance globally while preserving recovery-only world-up forces."""
    cfg = UnitreeG123AmpEnvCfg().events.recovery_assist
    cfg.params.update(initial_force=400.0, start_steps=100, decay_steps=100)
    cfg.params["asset_cfg"].body_ids = [1]
    composer = SimpleNamespace(set_forces_and_torques_index=Mock())
    asset = SimpleNamespace(permanent_wrench_composer=composer)
    env = SimpleNamespace(
        scene={"robot": asset},
        num_envs=3,
        device="cpu",
        common_step_counter=0,
        amp_recovery_env_mask=torch.tensor([True, False, True]),
    )
    assist = cfg.func(cfg, env)
    for step, magnitude in ((0, 0.0), (99, 0.0), (100, 400.0), (150, 200.0), (200, 0.0), (250, 0.0)):
        env.common_step_counter = step
        assist(env, None, **cfg.params)
        applied = composer.set_forces_and_torques_index.call_args.kwargs
        expected = torch.zeros(3, 1, 3)
        expected[[0, 2], 0, 2] = magnitude
        torch.testing.assert_close(applied["forces"], expected)
        assert applied["is_global"] is True
        assert applied["body_ids"] == [1]
        assert assist.current_force == magnitude

    env.common_step_counter = 150
    assist.reset(torch.tensor([0]))
    applied = composer.set_forces_and_torques_index.call_args.kwargs
    torch.testing.assert_close(applied["forces"], torch.tensor([[[0.0, 0.0, 200.0]]]))
    env.amp_recovery_env_mask[0] = False
    assist.reset(torch.tensor([0]))
    applied = composer.set_forces_and_torques_index.call_args.kwargs
    torch.testing.assert_close(applied["forces"], torch.zeros(1, 1, 3))


def test_upright_reward_depends_on_height_and_survives_recovery_exit():
    """Encourage upright torsos near standing height, including after failure clears."""
    cfg = UnitreeG123AmpEnvCfg()
    term = cfg.rewards.track_body_upright
    body_names = ["pelvis", "torso_link"]
    for name in ("body_cfg", "height_cfg"):
        selector = term.params[name]
        selector.body_ids, _ = resolve_matching_names(selector.body_names, body_names)

    # Upright, forward 30/90/180 deg, sideways 90 deg, yaw 90 deg, low/high upright.
    quat = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.2588190451, 0.0, 0.9659258263],
            [0.0, 0.7071067812, 0.0, 0.7071067812],
            [0.0, 1.0, 0.0, 0.0],
            [0.7071067812, 0.0, 0.0, 0.7071067812],
            [0.0, 0.0, 0.7071067812, 0.7071067812],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    positions = torch.zeros(8, 2, 3)
    positions[:, 0, 2] = torch.tensor([0.8] * 6 + [0.2, 1.4])
    positions[:, 1, 2] = 1.0
    data = SimpleNamespace(
        body_link_quat_w=SimpleNamespace(torch=torch.stack((torch.roll(quat, 1, 0), quat), dim=1)),
        body_link_pos_w=SimpleNamespace(torch=positions),
        default_root_state=torch.tensor([[0.0, 0.0, 0.8]] * 8),
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        termination_manager=SimpleNamespace(
            _delay_env_mask=torch.ones(8, dtype=torch.bool),
            _delay_counters=torch.ones(8, dtype=torch.long),
        ),
    )
    reward = term.func(env, **term.params)
    assert reward.shape == (8,)
    assert torch.isfinite(reward).all()
    assert torch.all((reward >= 0.0) & (reward <= 1.0))
    assert reward[0] > reward[1] > reward[2] > reward[3]
    torch.testing.assert_close(reward[[0, 5]], torch.ones(2))
    torch.testing.assert_close(reward[2], reward[4])
    torch.testing.assert_close(reward[6:], torch.tensor([0.01831564, 0.01831564]))

    # Losing the pending-failure flag must not remove posture guidance.
    env.termination_manager._delay_counters.zero_()
    torch.testing.assert_close(term.func(env, **term.params), reward)
    env.termination_manager = SimpleNamespace()
    torch.testing.assert_close(term.func(env, **term.params), reward)


def test_velocity_curriculum_reaches_limits_without_compounding():
    """Expand both endpoints by training progress, regardless of reset frequency."""
    cfg = UnitreeG123AmpEnvCfg()
    # Use fixed endpoints to test interpolation independently of task tuning.
    cfg.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
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
    term.cfg.ranges.lin_vel_x = (-0.5, 1.0)
    term.cfg.limit_ranges.lin_vel_x = (-2.0, 4.0)
    custom_cfg.curriculum.command_vel.params["num_steps"] = 100
    curriculum = custom_cfg.curriculum.command_vel
    update = curriculum.func(curriculum, env)
    env.common_step_counter = 50
    update(env, [0], **curriculum.params)
    assert term.cfg.ranges.lin_vel_x == (-1.25, 2.5)
