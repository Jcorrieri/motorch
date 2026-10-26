"""Opt-in benchmark for MoTorch training scale.

Run with ``MOTORCH_RUN_SCALE_TEST=1 uv run pytest -s tests/test_training_scale.py``.
Set ``MOTORCH_SCALE_WIDTHS`` to a comma-separated width list to change the sweep.
Set ``MOTORCH_SCALE_DEPTHS`` to change the hidden-layer count sweep.
"""

import os
import time

import numpy as np
import pytest

import motorch.nn as nn
from motorch.nn.modules.loss import LogisticLoss
from motorch.optim.sgd import SGD
from motorch.tensor import tensor


class BenchmarkMLP(nn.Module):
    """MLP used to compare training cost as width and depth grow."""

    def __init__(self, input_size, hidden_size, hidden_layer_count):
        super().__init__()
        self.hidden_layers = []
        self.activations = []
        for index in range(hidden_layer_count):
            layer_input_size = input_size if index == 0 else hidden_size
            layer = nn.Linear(layer_input_size, hidden_size)
            activation = nn.ReLU()
            setattr(self, f"hidden_layer{index}", layer)
            setattr(self, f"activation{index}", activation)
            self.hidden_layers.append(layer)
            self.activations.append(activation)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        for layer, activation in zip(self.hidden_layers, self.activations):
            inputs = activation(layer(inputs))
        return self.output(inputs)


def _widths_from_environment():
    configured_widths = os.getenv("MOTORCH_SCALE_WIDTHS", "64,128,256,512,1024")
    return [int(width.strip()) for width in configured_widths.split(",")]


def _depths_from_environment():
    configured_depths = os.getenv("MOTORCH_SCALE_DEPTHS", "1,2,4,8")
    return [int(depth.strip()) for depth in configured_depths.split(",")]


@pytest.mark.skipif(
    os.getenv("MOTORCH_RUN_SCALE_TEST") != "1",
    reason="set MOTORCH_RUN_SCALE_TEST=1 to run the training scale benchmark",
)
def test_training_scale_benchmark():
    """Print parameter, estimated memory, and training-time measurements."""
    rng = np.random.default_rng(0)
    input_size = 64
    batch_size = int(os.getenv("MOTORCH_SCALE_BATCH_SIZE", "32"))
    steps = int(os.getenv("MOTORCH_SCALE_STEPS", "50"))
    inputs = tensor(rng.normal(size=(batch_size, input_size)))
    labels = tensor(rng.choice([-1.0, 1.0], size=(batch_size, 1)))
    loss_function = LogisticLoss()

    print("depth | width | parameters | parameter + gradient storage | ms / step")
    for hidden_layer_count in _depths_from_environment():
        for hidden_size in _widths_from_environment():
            model = BenchmarkMLP(input_size, hidden_size, hidden_layer_count)
            parameters = list(model.parameters())
            for parameter in parameters:
                parameter.data[:] = rng.normal(0.0, 0.05, size=parameter.shape)

            optimizer = SGD(parameters, lr=0.001)
            parameter_count = sum(parameter.data.size for parameter in parameters)
            storage_bytes = sum(
                parameter.data.nbytes * 2 for parameter in parameters
            )

            start_time = time.perf_counter()
            for _ in range(steps):
                optimizer.zero_grad()
                loss = loss_function(model(inputs), labels)
                loss.backward()
                optimizer.step()
            elapsed = time.perf_counter() - start_time

            assert np.isfinite(loss.data).all()
            assert all(parameter.grad is not None for parameter in parameters)
            print(
                f"{hidden_layer_count:>5} | {hidden_size:>5} | "
                f"{parameter_count:>10,} | "
                f"{storage_bytes / 1024**2:>10.2f} MiB | "
                f"{elapsed / steps * 1000:>8.2f}"
            )
