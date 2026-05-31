"""
Integration tests for a multi-layer MoTorch MLP.

These tests treat the network as a black box and verify correctness through:
  - Structural invariants (shapes, graph construction)
  - Behavioral contracts (loss decreases, gradients flow, weights update)
  - Numerical gradient checks (finite differences vs analytical)
  - Known failure modes (gradient accumulation, zero_grad, stale graph)

No expected gradient values are computed by hand — correctness is established
through finite differences and behavioral properties.
"""

import numpy as np
import pytest
from motorch.tensor import tensor, Tensor
from motorch.nn.modules.linear import Linear
from motorch.nn.modules.activations import AltSigmoid, Sigmoid
from motorch.nn.modules.loss import LogisticLoss
from motorch.optim.sgd import SGD
import motorch.nn as nn


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_mlp(layer_sizes, activation=AltSigmoid):
    """Build a simple MLP with the given layer sizes and activation."""

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = []
            self.acts   = []
            for i, (in_f, out_f) in enumerate(zip(layer_sizes, layer_sizes[1:])):
                layer = Linear(in_f, out_f)
                act   = activation()
                setattr(self, f"layer{i}", layer)
                setattr(self, f"act{i}",   act)
                self.layers.append(layer)
                self.acts.append(act)

        def forward(self, x):
            for layer, act in zip(self.layers, self.acts):
                x = act(layer(x))
            return x

    return MLP()


def set_weights(model, seed=42):
    """Fill all parameters with small deterministic values."""
    rng = np.random.default_rng(seed)
    for param in model.parameters():
        param.data = rng.normal(0, 0.1, param.shape)


def make_data(n=20, n_features=4, seed=0):
    """Simple linearly separable binary dataset with labels in {+1, -1}."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, n_features))
    y = np.where(X[:, 0] > 0, 1.0, -1.0)
    return tensor(X), tensor(y.reshape(-1, 1))


# ── output shape ──────────────────────────────────────────────────────────────

class TestOutputShape:

    def test_single_sample_output_shape(self):
        model = make_mlp([4, 8, 1])
        set_weights(model)
        x = tensor(np.ones((1, 4)))
        out = model(x)
        assert out.shape == (1, 1)

    def test_batch_output_shape(self):
        model = make_mlp([4, 8, 1])
        set_weights(model)
        x = tensor(np.ones((16, 4)))
        out = model(x)
        assert out.shape == (16, 1)

    def test_deep_network_output_shape(self):
        model = make_mlp([4, 16, 8, 4, 1])
        set_weights(model)
        x = tensor(np.ones((8, 4)))
        out = model(x)
        assert out.shape == (8, 1)

    def test_altsigmoid_output_range(self):
        model = make_mlp([4, 8, 1], activation=AltSigmoid)
        set_weights(model)
        x = tensor(np.random.randn(50, 4))
        out = model(x)
        assert np.all(out.data > -1.0)
        assert np.all(out.data <  1.0)

    def test_returns_tensor(self):
        model = make_mlp([4, 8, 1])
        set_weights(model)
        out = model(tensor(np.ones((1, 4))))
        assert isinstance(out, Tensor)


# ── graph construction ────────────────────────────────────────────────────────

class TestGraphConstruction:

    def test_grad_fn_set_on_output(self):
        model = make_mlp([4, 8, 1])
        set_weights(model)
        x = tensor(np.ones((2, 4)), requires_grad=True)
        out = model(x)
        assert out.grad_fn is not None

    def test_all_parameters_have_grad_after_backward(self):
        model = make_mlp([4, 8, 1])
        set_weights(model)
        X, y = make_data()
        loss_fn = LogisticLoss()
        loss = loss_fn(model(X), y)
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None, f"Parameter {param.shape} has no grad"

    def test_grad_shapes_match_parameter_shapes(self):
        model = make_mlp([4, 8, 1])
        set_weights(model)
        X, y = make_data()
        loss_fn = LogisticLoss()
        loss_fn(model(X), y).backward()
        for param in model.parameters():
            assert param.grad.shape == param.shape, \
                f"Grad shape {param.grad.shape} != param shape {param.shape}"

    def test_loss_scalar_output(self):
        model = make_mlp([4, 8, 1])
        set_weights(model)
        X, y = make_data()
        loss = LogisticLoss()(model(X), y)
        assert loss.shape == ()


# ── gradient flow ─────────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_gradients_nonzero(self):
        """All parameter gradients should be non-zero for a typical input."""
        model = make_mlp([4, 8, 1])
        set_weights(model)
        X, y = make_data()
        LogisticLoss()(model(X), y).backward()
        for param in model.parameters():
            assert np.any(param.grad.data != 0), \
                f"All-zero gradient for parameter of shape {param.shape}"

    def test_numerical_gradient_check(self):
        """
        Finite difference check on a subset of parameters.
        This is the primary correctness test — no reimplementation of grad logic.
        """
        model = make_mlp([2, 3, 1])
        set_weights(model, seed=1)
        X = tensor([[1.0, -0.5]])
        y = tensor([[1.0]])
        loss_fn = LogisticLoss()
        eps = 1e-5

        # Analytical gradients
        loss_fn(model(X), y).backward()

        # Check a sample of parameters against finite differences
        for param in model.parameters():
            analytical = param.grad.data.copy()
            flat = param.data.flatten()
            numerical  = np.zeros_like(flat)

            for i in range(len(flat)):
                original = flat[i]

                param.data.flat[i] = original + eps
                loss_plus = float(loss_fn(model(X), y).data)

                param.data.flat[i] = original - eps
                loss_minus = float(loss_fn(model(X), y).data)

                numerical[i] = (loss_plus - loss_minus) / (2 * eps)
                param.data.flat[i] = original  # restore

            np.testing.assert_allclose(
                analytical.flatten(), numerical,
                rtol=1e-3, atol=1e-5,
                err_msg=f"Numerical grad mismatch for param shape {param.shape}"
            )

    def test_gradients_flow_to_first_layer(self):
        """Gradients must reach the earliest layer — catches vanishing/blocked flow."""
        model = make_mlp([4, 8, 4, 1])
        set_weights(model)
        X, y = make_data()
        LogisticLoss()(model(X), y).backward()
        first_layer = model.layer0
        assert np.any(first_layer.weight.grad.data != 0)
        assert np.any(first_layer.bias.grad.data   != 0)


# ── training dynamics ─────────────────────────────────────────────────────────

class TestTrainingDynamics:

    def test_loss_decreases_over_steps(self):
        """Loss should decrease monotonically over a few steps on a simple dataset."""
        model = make_mlp([4, 8, 1])
        set_weights(model)
        X, y = make_data(n=50)
        loss_fn   = LogisticLoss()
        optimizer = SGD(model.parameters(), lr=0.5)
        losses = []

        for _ in range(10):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.data))

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_weights_change_after_step(self):
        model = make_mlp([4, 8, 1])
        set_weights(model)
        X, y = make_data()
        loss_fn   = LogisticLoss()
        optimizer = SGD(model.parameters(), lr=0.1)

        weights_before = [p.data.copy() for p in model.parameters()]
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        optimizer.step()
        weights_after = [p.data.copy() for p in model.parameters()]

        for before, after in zip(weights_before, weights_after):
            assert not np.allclose(before, after), "Weights unchanged after optimizer step"

    def test_weights_unchanged_without_step(self):
        model = make_mlp([4, 8, 1])
        set_weights(model)
        X, y = make_data()

        weights_before = [p.data.copy() for p in model.parameters()]
        LogisticLoss()(model(X), y).backward()  # no optimizer.step()
        weights_after = [p.data.copy() for p in model.parameters()]

        for before, after in zip(weights_before, weights_after):
            assert np.allclose(before, after), "Weights changed without optimizer step"

    def test_zero_lr_no_weight_change(self):
        model = make_mlp([4, 8, 1])
        set_weights(model)
        X, y = make_data()
        optimizer = SGD(model.parameters(), lr=0.0)

        weights_before = [p.data.copy() for p in model.parameters()]
        optimizer.zero_grad()
        LogisticLoss()(model(X), y).backward()
        optimizer.step()
        weights_after = [p.data.copy() for p in model.parameters()]

        for before, after in zip(weights_before, weights_after):
            assert np.allclose(before, after), "Weights changed with lr=0"


# ── multi-pass correctness ────────────────────────────────────────────────────

class TestMultiPass:

    def test_gradients_consistent_across_passes(self):
        """Same input should produce same gradients on every pass after zero_grad."""
        model = make_mlp([4, 8, 1])
        set_weights(model)
        X, y = make_data(n=10)
        loss_fn   = LogisticLoss()
        optimizer = SGD(model.parameters(), lr=0.0)  # lr=0: weights stay fixed

        grads_per_pass = []
        for _ in range(3):
            optimizer.zero_grad()
            loss_fn(model(X), y).backward()
            grads_per_pass.append([p.grad.data.copy() for p in model.parameters()])

        for param_idx in range(len(grads_per_pass[0])):
            np.testing.assert_allclose(
                grads_per_pass[0][param_idx],
                grads_per_pass[1][param_idx],
                rtol=1e-6, atol=1e-8,
                err_msg="Gradients differ between pass 1 and pass 2"
            )
            np.testing.assert_allclose(
                grads_per_pass[1][param_idx],
                grads_per_pass[2][param_idx],
                rtol=1e-6, atol=1e-8,
                err_msg="Gradients differ between pass 2 and pass 3"
            )

    def test_gradients_do_not_accumulate_across_passes(self):
        """Gradients after two passes with zero_grad should equal one-pass grads."""
        model = make_mlp([4, 8, 1])
        set_weights(model)
        X, y = make_data(n=10)
        loss_fn   = LogisticLoss()
        optimizer = SGD(model.parameters(), lr=0.0)

        # Single pass
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        single_pass_grads = [p.grad.data.copy() for p in model.parameters()]

        # Second pass — grads should match, not double
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        second_pass_grads = [p.grad.data.copy() for p in model.parameters()]

        for g1, g2 in zip(single_pass_grads, second_pass_grads):
            np.testing.assert_allclose(g1, g2, rtol=1e-6, atol=1e-8,
                err_msg="Gradients accumulated across passes — zero_grad may be broken")

    def test_loss_monotonically_decreasing_multi_pass(self):
        """Loss should decrease on every step, not just overall."""
        model = make_mlp([4, 8, 1])
        set_weights(model, seed=7)
        X, y = make_data(n=100, seed=7)
        loss_fn   = LogisticLoss()
        optimizer = SGD(model.parameters(), lr=0.3)
        prev_loss = float("inf")

        for step in range(15):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            current_loss = float(loss.data)
            assert current_loss < prev_loss, \
                f"Loss increased at step {step}: {prev_loss:.4f} -> {current_loss:.4f}"
            prev_loss = current_loss

    def test_grad_fn_cleared_after_backward(self):
        """grad_fn should be None on all nodes after backward completes."""
        model = make_mlp([4, 8, 1])
        set_weights(model)
        X, y = make_data(n=5)
        loss_fn = LogisticLoss()
        loss = loss_fn(model(X), y)

        from motorch.autograd.topological_sort import topological_sort
        nodes = topological_sort(loss)
        loss.backward()
        for node in nodes:
            assert node.grad_fn is None, \
                f"grad_fn not cleared on node of shape {node.shape}"
