"""CUDA graph replay for fixed-shape learner updates."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from tensordict import TensorDict
from torch import nn


def graph_state(modules: Iterable[nn.Module], optimizers: Iterable[torch.optim.Optimizer]) -> list[torch.Tensor]:
    """Return module and optimizer storage referenced by the captured update."""
    state = [tensor for module in modules for tensor in (*module.parameters(), *module.buffers())]
    for optimizer in optimizers:
        state.extend(
            tensor for entry in optimizer.state.values() for tensor in entry.values() if torch.is_tensor(tensor)
        )
        state.extend(group["lr"] for group in optimizer.param_groups if torch.is_tensor(group["lr"]))
    return state


class CapturedUpdate:
    """Warm up an update, then replay it against persistent CUDA input storage."""

    def __init__(
        self,
        update: Callable[[TensorDict], dict[str, torch.Tensor]],
        state: Callable[[], list[torch.Tensor]],
        warmup: int = 3,
    ) -> None:
        self._update = update
        self._state = state
        self._warmup = warmup
        self._inputs: TensorDict | None = None
        self._stream: torch.cuda.Stream | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._storage: list[int] = []
        self._outputs: dict[str, torch.Tensor] = {}

    def __call__(self, inputs: TensorDict) -> dict[str, torch.Tensor]:
        if not inputs.is_cuda:
            return self._update(inputs)

        with torch.cuda.device(inputs.device):
            current = torch.cuda.current_stream()
            if self._inputs is None:
                self._inputs = inputs.clone()
                self._stream = torch.cuda.Stream()
            else:
                self._inputs.update_(inputs, non_blocking=True)

            stream = self._stream
            stream.wait_stream(current)
            if self._warmup:
                self._warmup -= 1
                with torch.cuda.stream(stream):
                    outputs = self._update(self._inputs)
                current.wait_stream(stream)
                return outputs

            if self._graph is None:
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph, stream=stream):
                    self._outputs = self._update(self._inputs)
                self._storage = [tensor.data_ptr() for tensor in self._state()]
            elif self._storage != [tensor.data_ptr() for tensor in self._state()]:
                raise RuntimeError("Learner tensor storage changed after CUDA graph capture.")
            self._graph.replay()
            return {name: value.clone() for name, value in self._outputs.items()}
