"""Tests for the real motorch Linear module."""

import numpy as np

import motorch as mo
from motorch.nn import Linear


def set_linear_params(layer, weight, bias):
    layer.weight.data = np.array(weight, dtype=float)
    layer.bias.data = np.array(bias, dtype=float)


class TestLinearForward:
    def test_parameters_have_expected_shapes(self):
        layer = Linear(3, 2)

        assert layer.weight.shape == (3, 2)
        assert layer.bias.shape == (1, 2)

    def test_single_sample_forward_values(self):
        layer = Linear(2, 2)
        set_linear_params(
            layer,
            weight=[[1.0, -2.0], [3.0, 4.0]],
            bias=[[0.5, -1.5]],
        )

        out = layer(mo.tensor([[2.0, -1.0]]))

        np.testing.assert_allclose(out.data, [[-0.5, -9.5]])

    def test_batch_forward_values_and_shape(self):
        layer = Linear(2, 3)
        set_linear_params(
            layer,
            weight=[[1.0, 0.0, -1.0], [2.0, 3.0, 0.5]],
            bias=[[0.25, -0.5, 1.0]],
        )
        x = mo.tensor([[1.0, 2.0], [-1.0, 0.5], [0.0, -2.0]])

        out = layer(x)

        assert out.shape == (3, 3)
        np.testing.assert_allclose(
            out.data, x.data @ layer.weight.data + layer.bias.data
        )


class TestLinearGradients:
    def test_gradients_match_matrix_calculus_for_batched_upstream_grad(self):
        layer = Linear(2, 3)
        set_linear_params(
            layer,
            weight=[[1.0, -2.0, 0.5], [3.0, 4.0, -1.0]],
            bias=[[0.25, -0.5, 1.0]],
        )
        x = mo.tensor([[2.0, -1.0], [0.0, 3.0], [-4.0, 1.5]])
        upstream = mo.tensor([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0], [-1.5, 0.0, 2.0]])

        out = layer(x)
        out.grad = upstream
        out.grad_fn()

        np.testing.assert_allclose(x.grad.data, upstream.data @ layer.weight.data.T)
        np.testing.assert_allclose(layer.weight.grad.data, x.data.T @ upstream.data)
        np.testing.assert_allclose(
            layer.bias.grad.data,
            upstream.data.sum(axis=0, keepdims=True),
        )
        assert layer.bias.grad.shape == layer.bias.shape

    def test_bias_gradient_sums_over_batch_not_features(self):
        layer = Linear(2, 2)
        set_linear_params(
            layer,
            weight=[[0.5, -1.0], [2.0, 3.0]],
            bias=[[0.0, 0.0]],
        )
        x = mo.tensor([[1.0, 2.0], [3.0, 4.0], [-2.0, 0.5]])
        upstream = mo.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        out = layer(x)
        out.grad = upstream
        out.grad_fn()

        np.testing.assert_allclose(layer.bias.grad.data, [[6.0, 60.0]])

    def test_weight_and_bias_gradients_match_finite_differences(self):
        layer = Linear(2, 2)
        set_linear_params(
            layer,
            weight=[[0.75, -1.25], [2.0, 0.5]],
            bias=[[0.1, -0.3]],
        )
        x = mo.tensor([[1.5, -2.0], [0.25, 3.0]])
        upstream = mo.tensor([[0.5, -1.5], [2.0, 0.75]])
        eps = 1e-6

        out = layer(x)
        out.grad = upstream
        out.grad_fn()

        def objective():
            return float(np.sum(layer(x).data * upstream.data))

        for param in [layer.weight, layer.bias]:
            analytical = param.grad.data.copy()
            numerical = np.zeros_like(param.data)

            for index in np.ndindex(param.shape):
                original = param.data[index]
                param.data[index] = original + eps
                loss_plus = objective()
                param.data[index] = original - eps
                loss_minus = objective()
                numerical[index] = (loss_plus - loss_minus) / (2 * eps)
                param.data[index] = original

            np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-7)
