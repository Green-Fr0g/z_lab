"""Muon and Adam optimizer construction for SuperMimic learners."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn


def build_optimizers(
    modules: Sequence[nn.Module],
    learning_rate: float,
    device: torch.device,
    betas: tuple[float, float] = (0.9, 0.999),
) -> list[torch.optim.Optimizer]:
    """Use Muon for hidden linear weights and Adam for the remaining parameters."""
    muon_parameters = [
        layer.weight
        for module in modules
        for sequential in module.modules()
        if isinstance(sequential, nn.Sequential)
        for layer in [child for child in sequential if isinstance(child, nn.Linear)][1:-1]
    ]
    muon_parameter_ids = {id(parameter) for parameter in muon_parameters}
    adam_parameters = [
        parameter for module in modules for parameter in module.parameters() if id(parameter) not in muon_parameter_ids
    ]
    if device.type == "cuda":
        adam = torch.optim.Adam(
            adam_parameters,
            lr=torch.tensor(learning_rate, device=device),
            betas=betas,
            fused=True,
            capturable=True,
        )
    else:
        adam = torch.optim.Adam(adam_parameters, lr=learning_rate, betas=betas)
    optimizers: list[torch.optim.Optimizer] = [adam]
    if muon_parameters:
        optimizers.append(torch.optim.Muon(muon_parameters, lr=learning_rate, adjust_lr_fn="match_rms_adamw"))
    return optimizers


def optimize_step(
    optimizers: Iterable[torch.optim.Optimizer],
    loss: torch.Tensor,
    parameters: Iterable[torch.Tensor],
    max_grad_norm: float,
) -> torch.Tensor:
    """Backpropagate one loss, clip gradients and step each optimizer."""
    optimizers = list(optimizers)
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    for optimizer in optimizers:
        optimizer.step()
    return grad_norm
