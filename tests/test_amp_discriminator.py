"""Numerical contracts for the AMP discriminator reward and gradient penalty."""

import pytest
import torch
from rsl_rl.models import AMPDiscriminator

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def _fixed_logit_discriminator(logit: float, amp_reward_coef: float, task_reward_lerp: float = 0.0):
    """Build a discriminator whose output is a constant, so the reward formula is an exact number."""
    discriminator = AMPDiscriminator(
        input_dim=4,
        amp_reward_coef=amp_reward_coef,
        hidden_layer_sizes=[8],
        device="cpu",
        task_reward_lerp=task_reward_lerp,
    )
    with torch.no_grad():
        discriminator.amp_linear.weight.zero_()
        discriminator.amp_linear.bias.fill_(logit)
    return discriminator


def test_reward_uses_the_quadratic_amp_form_and_clamps_at_zero():
    state = torch.zeros(3, 2)
    next_state = torch.zeros(3, 2)

    # logit 1 is the expert target, so the unclamped imitation reward equals the coefficient.
    reward, logits = _fixed_logit_discriminator(1.0, amp_reward_coef=2.0).reward(state, next_state)
    torch.testing.assert_close(logits, torch.ones(3))
    torch.testing.assert_close(reward, torch.full((3,), 2.0))

    # logit -3 gives 1 - 0.25 * 16 = -3, which the AMP reward clamps to zero.
    reward, _ = _fixed_logit_discriminator(-3.0, amp_reward_coef=2.0).reward(state, next_state)
    torch.testing.assert_close(reward, torch.zeros(3))


def test_task_reward_lerp_requires_and_blends_the_task_terminal_reward():
    state = torch.zeros(2, 2)
    next_state = torch.zeros(2, 2)
    task_reward = torch.tensor([1.0, -1.0])
    discriminator = _fixed_logit_discriminator(1.0, amp_reward_coef=2.0, task_reward_lerp=0.25)

    with pytest.raises(ValueError, match="task_reward"):
        discriminator.reward(state, next_state)

    reward, _ = discriminator.reward(state, next_state, task_reward=task_reward)
    torch.testing.assert_close(reward, 0.75 * torch.tensor([2.0, 2.0]) + 0.25 * task_reward)


def test_gradient_penalty_is_smooth_and_does_not_leak_gradients_into_inputs():
    discriminator = AMPDiscriminator(
        input_dim=4, amp_reward_coef=1.0, hidden_layer_sizes=[4], device="cpu", activation="relu"
    )
    expert_state = torch.randn(5, 2)
    expert_next_state = torch.randn(5, 2)

    penalty = discriminator.compute_gradient_penalty(
        torch.cat([expert_state, expert_next_state], dim=-1), lambda_=10.0
    )
    assert penalty.ndim == 0 and penalty.item() >= 0.0
    penalty.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in discriminator.parameters()
    )
    assert not expert_state.requires_grad and not expert_next_state.requires_grad
