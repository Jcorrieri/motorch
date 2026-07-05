"""Tests for motorch activation modules."""

import numpy as np
import pytest
from motorch.tensor import tensor, Tensor
from motorch.nn.modules.activations import Sgn, Sigmoid, AltSigmoid


# ── helpers ───────────────────────────────────────────────────────────────────


def assert_close(actual, expected, rtol=1e-5, atol=1e-7):
    np.testing.assert_allclose(
        actual.data if isinstance(actual, Tensor) else np.array(actual),
        expected if isinstance(expected, np.ndarray) else np.array(expected),
        rtol=rtol,
        atol=atol,
    )


def numerical_grad(module, x_val, eps=1e-5):
    """Central finite difference gradient estimate for a scalar module."""
    out_plus = float(module(tensor(float(x_val + eps))).data)
    out_minus = float(module(tensor(float(x_val - eps))).data)
    return (out_plus - out_minus) / (2 * eps)


# ── Sgn ───────────────────────────────────────────────────────────────────────


class TestSgn:
    def setup_method(self):
        self.sgn = Sgn()

    def test_positive_input(self):
        assert_close(self.sgn(tensor(1.0)), 1.0)

    def test_negative_input(self):
        assert_close(self.sgn(tensor(-1.0)), -1.0)

    def test_zero_input(self):
        # zero is non-negative so should return +1
        assert_close(self.sgn(tensor(0.0)), 1.0)

    def test_vector_input(self):
        x = tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        out = self.sgn(x)
        assert_close(out, [-1.0, -1.0, 1.0, 1.0, 1.0])

    def test_output_values_only_plus_minus_one(self):
        x = tensor(np.linspace(-10, 10, 100))
        out = self.sgn(x)
        unique = np.unique(out.data)
        assert set(unique).issubset({-1.0, 1.0})

    def test_returns_tensor(self):
        assert isinstance(self.sgn(tensor(1.0)), Tensor)

    def test_no_grad_fn(self):
        # Sgn is non-differentiable — no grad_fn should be set
        x = tensor(1.0, requires_grad=True)
        out = self.sgn(x)
        assert out.grad_fn is None


# ── Sigmoid ───────────────────────────────────────────────────────────────────


class TestSigmoid:
    def setup_method(self):
        self.sigmoid = Sigmoid()

    # forward values

    def test_zero_input(self):
        assert_close(self.sigmoid(tensor(0.0)), 0.5)

    def test_positive_input(self):
        expected = 1 / (1 + np.exp(-1.0))
        assert_close(self.sigmoid(tensor(1.0)), expected)

    def test_negative_input(self):
        expected = 1 / (1 + np.exp(1.0))
        assert_close(self.sigmoid(tensor(-1.0)), expected)

    def test_output_range(self):
        x = tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        out = self.sigmoid(x)
        assert np.all(out.data > 0.0)
        assert np.all(out.data < 1.0)

    def test_vector_input(self):
        vals = np.array([-2.0, 0.0, 2.0])
        expected = 1 / (1 + np.exp(-vals))
        assert_close(self.sigmoid(tensor(vals)), expected)

    def test_returns_tensor(self):
        assert isinstance(self.sigmoid(tensor(1.0)), Tensor)

    # graph construction

    def test_grad_fn_set_when_requires_grad(self):
        x = tensor(1.0, requires_grad=True)
        out = self.sigmoid(x)
        assert out.grad_fn is not None

    def test_grad_fn_none_when_no_grad(self):
        x = tensor(1.0, requires_grad=False)
        out = self.sigmoid(x)
        assert out.grad_fn is None

    def test_children_set(self):
        x = tensor(1.0, requires_grad=True)
        out = self.sigmoid(x)
        assert any(c is x for c in out._children)

    # gradient correctness

    def test_grad_zero_input(self):
        # sigmoid_grad(0) = 0.5 * 0.5 = 0.25
        x = tensor(0.0, requires_grad=True)
        out = self.sigmoid(x)
        out.backward()
        assert_close(x.grad, 0.25)

    def test_grad_known_value(self):
        x_val = 1.0
        s = 1 / (1 + np.exp(-x_val))
        expected_grad = s * (1 - s)
        x = tensor(x_val, requires_grad=True)
        self.sigmoid(x).backward()
        assert_close(x.grad, expected_grad)

    def test_grad_matches_numerical(self):
        for x_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            x = tensor(x_val, requires_grad=True)
            self.sigmoid(x).backward()
            analytical = float(x.grad.item()) if x.grad is not None else 0.0
            numerical = numerical_grad(self.sigmoid, x_val)
            assert analytical == pytest.approx(numerical, rel=1e-4), (
                f"Grad mismatch at x={x_val}: analytical={analytical}, numerical={numerical}"
            )

    def test_grad_does_not_accumulate_across_calls(self):
        x = tensor(1.0, requires_grad=True)
        out1 = self.sigmoid(x)
        out1.backward()
        grad_after_first = float(x.grad.item()) if x.grad is not None else 0.0
        x.grad = None
        out2 = self.sigmoid(x)
        out2.backward()
        grad_after_second = float(x.grad.item()) if x.grad is not None else 0.0
        assert grad_after_first == pytest.approx(grad_after_second, rel=1e-6)


# ── AltSigmoid ────────────────────────────────────────────────────────────────


class TestAltSigmoid:
    def setup_method(self):
        self.altsig = AltSigmoid()

    # forward values

    def test_zero_input(self):
        # 2 * sigmoid(0) - 1 = 2 * 0.5 - 1 = 0.0
        assert_close(self.altsig(tensor(0.0)), 0.0)

    def test_positive_input(self):
        x_val = 1.0
        expected = 2 * (1 / (1 + np.exp(-x_val))) - 1
        assert_close(self.altsig(tensor(x_val)), expected)

    def test_negative_input(self):
        x_val = -1.0
        expected = 2 * (1 / (1 + np.exp(-x_val))) - 1
        assert_close(self.altsig(tensor(x_val)), expected)

    def test_output_range(self):
        x = tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        out = self.altsig(x)
        assert np.all(out.data > -1.0)
        assert np.all(out.data < 1.0)

    def test_antisymmetry(self):
        # AltSigmoid is antisymmetric: f(-x) = -f(x)
        x_val = 1.5
        pos = float(self.altsig(tensor(x_val)).data)
        neg = float(self.altsig(tensor(-x_val)).data)
        assert pos == pytest.approx(-neg, rel=1e-6)

    def test_vector_input(self):
        vals = np.array([-2.0, 0.0, 2.0])
        expected = 2 / (1 + np.exp(-vals)) - 1
        assert_close(self.altsig(tensor(vals)), expected)

    def test_returns_tensor(self):
        assert isinstance(self.altsig(tensor(1.0)), Tensor)

    def test_output_rescaled_from_sigmoid(self):
        # AltSigmoid output should equal 2*Sigmoid(x) - 1
        sig = Sigmoid()
        x = tensor(1.5, requires_grad=False)
        expected = 2 * float(sig(x).data) - 1
        assert_close(self.altsig(tensor(1.5)), expected)

    # graph construction

    def test_grad_fn_set_when_requires_grad(self):
        x = tensor(1.0, requires_grad=True)
        out = self.altsig(x)
        assert out.grad_fn is not None

    def test_grad_fn_none_when_no_grad(self):
        x = tensor(1.0, requires_grad=False)
        out = self.altsig(x)
        assert out.grad_fn is None

    def test_children_set(self):
        x = tensor(1.0, requires_grad=True)
        out = self.altsig(x)
        assert any(c is x for c in out._children)

    # gradient correctness

    def test_grad_zero_input(self):
        # d/dx [2*sigmoid(x) - 1] at x=0 = 2 * 0.25 = 0.5
        x = tensor(0.0, requires_grad=True)
        self.altsig(x).backward()
        assert_close(x.grad, 0.5)

    def test_grad_known_value(self):
        x_val = 1.0
        s = 1 / (1 + np.exp(-x_val))
        expected_grad = 2 * s * (1 - s)
        x = tensor(x_val, requires_grad=True)
        self.altsig(x).backward()
        assert_close(x.grad, expected_grad)

    def test_grad_matches_numerical(self):
        for x_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            x = tensor(x_val, requires_grad=True)
            self.altsig(x).backward()
            analytical = float(x.grad.item()) if x.grad is not None else 0.0
            numerical = numerical_grad(self.altsig, x_val)
            assert analytical == pytest.approx(numerical, rel=1e-4), (
                f"Grad mismatch at x={x_val}: analytical={analytical}, numerical={numerical}"
            )

    def test_grad_double_sigmoid_grad(self):
        # AltSigmoid grad should be exactly 2x Sigmoid grad
        sig = Sigmoid()
        x_val = 1.5

        x1 = tensor(x_val, requires_grad=True)
        sig(x1).backward()
        sig_grad = float(x1.grad.item()) if x1.grad is not None else 0.0

        x2 = tensor(x_val, requires_grad=True)
        self.altsig(x2).backward()
        altsig_grad = float(x2.grad.item()) if x2.grad is not None else 0.0

        assert altsig_grad == pytest.approx(2 * sig_grad, rel=1e-6)

    def test_grad_does_not_accumulate_across_calls(self):
        x = tensor(1.0, requires_grad=True)
        self.altsig(x).backward()
        grad_after_first = float(x.grad.item()) if x.grad is not None else 0.0
        x.grad = None
        self.altsig(x).backward()
        grad_after_second = float(x.grad.item()) if x.grad is not None else 0.0
        assert grad_after_first == pytest.approx(grad_after_second, rel=1e-6)
