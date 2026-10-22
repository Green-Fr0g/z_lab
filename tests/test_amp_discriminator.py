"""Numerical contracts for temporal normalization, raw gradient penalties, and AMP rewards."""

from typing import cast

import pytest
import torch
import torch.nn as nn
from rsl_rl.models import AMPDiscriminator
from rsl_rl.modules import EmpiricalNormalization

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_discriminator_shares_frame_statistics_and_preserves_amp_formulas():
    discriminator = AMPDiscriminator(4, frame_dim=2, hidden_dims=[1], amp_reward_coef=2.0, task_reward_lerp=0.25)
    policy = torch.tensor([[1.0, 10.0, 3.0, 14.0]])
    expert = torch.tensor([[5.0, 18.0, 7.0, 22.0]])
    discriminator.update_normalization(policy, expert)
    normalizer = cast(EmpiricalNormalization, discriminator.obs_normalizer)
    input_layer = cast(nn.Linear, discriminator.mlp[0])
    head = cast(nn.Linear, discriminator.mlp[-1])
    # Four 2D frames: (1, 10), (3, 14), (5, 18), (7, 22).
    mean = torch.tensor([4.0, 16.0])
    variance = torch.tensor([5.0, 20.0])
    torch.testing.assert_close(normalizer.mean, mean)
    torch.testing.assert_close(normalizer.std.square(), variance)
    assert cast(torch.Tensor, normalizer.count).item() == 4

    with torch.no_grad():
        input_layer.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        input_layer.bias.fill_(100)
        head.weight.fill_(2)
        head.bias.fill_(-0.5)
    # Both temporal positions use the same statistics; the wide bias keeps the ReLU active.
    scaled = ((expert.reshape(2, 2) - mean) / (variance.sqrt() + 1e-2)).flatten()
    expected_logit = 2 * (scaled @ torch.tensor([1.0, 2.0, 3.0, 4.0]) + 100) - 0.5
    torch.testing.assert_close(discriminator(expert).squeeze(), expected_logit)
    # The optional discriminator clamp remains in effect with the built-in normalizer.
    outlier = torch.tensor([[1000.0, 16000.0, -1000.0, -16000.0]])
    torch.testing.assert_close(discriminator(outlier), torch.tensor([[119.5]]))
    # Raw input gradient is (2, 4, 6, 8), independently of the running normalizer.
    penalty = discriminator.compute_gradient_penalty(expert, lambda_=10)
    torch.testing.assert_close(penalty, torch.tensor(1200.0))
    penalty.backward()
    assert input_layer.weight.grad is not None and torch.isfinite(input_layer.weight.grad).all()
    assert not expert.requires_grad
    torch.testing.assert_close(expert, torch.tensor([[5.0, 18.0, 7.0, 22.0]]))

    # Identical inputs still have opposite least-squares targets for the two data sources.
    loss, metrics = discriminator.compute_loss(expert, expert)
    expert_loss = (expected_logit - 1).square()
    policy_loss = (expected_logit + 1).square()
    torch.testing.assert_close(metrics["amp_expert_loss"], expert_loss)
    torch.testing.assert_close(metrics["amp_policy_loss"], policy_loss)
    torch.testing.assert_close(metrics["amp_loss"], 0.5 * (expert_loss + policy_loss))
    torch.testing.assert_close(metrics["amp_grad_pen"], torch.tensor(1200.0))
    torch.testing.assert_close(loss, 0.5 * (expert_loss + policy_loss) + 1200)
    assert loss.requires_grad and all(not value.requires_grad for value in metrics.values())

    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    reward, prediction = discriminator.reward(policy, torch.tensor([4.0]))
    torch.testing.assert_close(prediction, torch.zeros(1))
    torch.testing.assert_close(reward, torch.tensor([2.125]))
    loss, metrics = discriminator.compute_loss(policy, expert, grad_pen_coef=0)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(metrics["amp_grad_pen"], torch.tensor(0.0))
