"""Tests for motorch autograd — DAG construction and gradient computation."""

import numpy as np
import pytest
from motorch.tensor import tensor, Tensor
from motorch import zeros_like


# ── helpers ──────────────────────────────────────────────────────────────────


def numerical_grad(f, x, eps=1e-5):
    """Central finite-difference gradient estimate for a scalar-output function."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def assert_grad_close(actual, expected, rtol=1e-4, atol=1e-6):
    np.testing.assert_allclose(
        actual.data if isinstance(actual, Tensor) else actual,
        expected.data if isinstance(expected, Tensor) else expected,
        rtol=rtol,
        atol=atol,
    )


# ── DAG node creation ─────────────────────────────────────────────────────────


class TestNodeCreation:
    def test_children_exist_when_left_requires_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=False)
        result = a + b
        assert len(result._children) > 0

    def test_children_exist_when_right_requires_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=False)
        b = Tensor([3.0, 4.0], requires_grad=True)
        result = a + b
        assert len(result._children) > 0

    def test_children_exist_when_both_require_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        result = a + b
        assert len(result._children) > 0

    def test_children_empty_when_no_inputs_require_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=False)
        b = Tensor([3.0, 4.0], requires_grad=False)
        result = a + b
        assert result._children == []

    def test_children_contain_inputs(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=False)
        result = a + b
        assert any(c is a for c in result._children)
        assert any(c is b for c in result._children)

    def test_scalar_input_wrapped_as_tensor_in_children(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        result = a + 2.0
        assert all(isinstance(c, Tensor) for c in result._children)

    def test_grad_fn_set_when_requires_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0])
        result = a + b
        assert result.grad_fn is not None

    def test_grad_fn_none_when_no_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=False)
        b = Tensor([3.0, 4.0], requires_grad=False)
        result = a + b
        assert result.grad_fn is None

    def test_requires_grad_propagates(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0])
        result = a + b
        assert result.requires_grad is True

    def test_requires_grad_false_when_no_inputs_require_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=False)
        b = Tensor([3.0, 4.0], requires_grad=False)
        result = a + b
        assert result.requires_grad is False


# ── version tracking ──────────────────────────────────────────────────────────


class TestVersionTracking:
    def test_version_increments_on_inplace(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        v = a._version
        a += 1.0
        assert a._version == v + 1

    def test_version_unchanged_on_out_of_place(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        v = a._version
        _ = a + 1.0
        assert a._version == v

    def test_version_increments_each_inplace(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        v = a._version
        a += 1.0
        a += 1.0
        a += 1.0
        assert a._version == v + 3

    def test_version_not_incremented_on_inplace_no_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=False)
        v = a._version
        a += 1.0
        assert a._version == v

    def test_inplace_raises_on_stale_version(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=True)
        z = x + y  # z captures x at version 0
        x += 1.0  # x mutated → version bumps
        z.grad = tensor(1.0)
        with pytest.raises(ValueError):
            z.grad_fn()  # should raise due to version mismatch


# ── single-op gradients ───────────────────────────────────────────────────────


class TestSingleOpGradients:
    # addition

    def test_add_grad_left(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0)
        z = x + y
        z.backward()
        assert_grad_close(x.grad, 1.0)

    def test_add_grad_right(self):
        x = tensor(2.0)
        y = tensor(3.0, requires_grad=True)
        z = x + y
        z.backward()
        assert_grad_close(y.grad, 1.0)

    def test_add_grad_both(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=True)
        z = x + y
        z.backward()
        assert_grad_close(x.grad, 1.0)
        assert_grad_close(y.grad, 1.0)

    def test_add_grad_scales_with_upstream(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=True)
        z = x + y
        z.grad = tensor(5.0)
        z.grad_fn()
        assert_grad_close(x.grad, 5.0)
        assert_grad_close(y.grad, 5.0)

    # multiplication

    def test_mul_grad_left(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(5.0)
        z = x * y
        z.backward()
        assert_grad_close(x.grad, 5.0)  # dz/dx = y

    def test_mul_grad_right(self):
        x = tensor(2.0)
        y = tensor(5.0, requires_grad=True)
        z = x * y
        z.backward()
        assert_grad_close(y.grad, 2.0)  # dz/dy = x

    def test_mul_grad_scales_with_upstream(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(5.0, requires_grad=True)
        z = x * y
        z.grad = tensor(3.0)
        z.grad_fn()
        assert_grad_close(x.grad, 15.0)  # y * upstream
        assert_grad_close(y.grad, 6.0)  # x * upstream

    # subtraction

    def test_sub_grad(self):
        x = tensor(4.0, requires_grad=True)
        y = tensor(1.0, requires_grad=True)
        z = x - y
        z.backward()
        assert_grad_close(x.grad, 1.0)
        assert_grad_close(y.grad, -1.0)

    # division

    def test_div_grad(self):
        x = tensor(6.0, requires_grad=True)
        y = tensor(2.0, requires_grad=True)
        z = x / y
        z.backward()
        assert_grad_close(x.grad, 0.5)  # 1/y
        assert_grad_close(y.grad, -1.5)  # -x/y²

    # matmul NOTE: Not yet implemented!

    # def test_matmul_grad(self):
    #     X = tensor([[1.0, 2.0]], requires_grad=True)   # (1,2)
    #     W = tensor([[3.0], [4.0]], requires_grad=True)  # (2,1)
    #     z = X @ W                                       # (1,1)
    #     z.backward()
    #     assert_grad_close(X.grad, [[3.0, 4.0]])         # dz/dX = W.T
    #     assert_grad_close(W.grad, [[1.0], [2.0]])        # dz/dW = X.T


# ── chained / multi-op gradients ─────────────────────────────────────────────


class TestChainedGradients:
    def test_add_chain(self):
        # w = (x + y) + v; dw/dx = dw/dy = dw/dv = 1
        x = tensor(1.0, requires_grad=True)
        y = tensor(2.0, requires_grad=True)
        v = tensor(3.0, requires_grad=True)
        w = (x + y) + v
        w.backward()
        assert_grad_close(x.grad, 1.0)
        assert_grad_close(y.grad, 1.0)
        assert_grad_close(v.grad, 1.0)

    def test_mul_chain(self):
        # w = (x * y) * v; dw/dx = y*v, dw/dy = x*v, dw/dv = x*y
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=True)
        v = tensor(4.0, requires_grad=True)
        w = (x * y) * v
        w.backward()
        assert_grad_close(x.grad, 12.0)  # y * v
        assert_grad_close(y.grad, 8.0)  # x * v
        assert_grad_close(v.grad, 6.0)  # x * y

    def test_mul_then_add(self):
        # w = x*y + v; dw/dx = y, dw/dy = x, dw/dv = 1
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=True)
        v = tensor(4.0, requires_grad=True)
        w = x * y + v
        w.backward()
        assert_grad_close(x.grad, 3.0)
        assert_grad_close(y.grad, 2.0)
        assert_grad_close(v.grad, 1.0)

    def test_add_then_mul(self):
        # w = (x + y) * v; dw/dx = v, dw/dy = v, dw/dv = x+y
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=True)
        v = tensor(4.0, requires_grad=True)
        w = (x + y) * v
        w.backward()
        assert_grad_close(x.grad, 4.0)
        assert_grad_close(y.grad, 4.0)
        assert_grad_close(v.grad, 5.0)

    def test_shared_input_accumulates_grad(self):
        # z = x * x; dz/dx = 2x
        x = tensor(3.0, requires_grad=True)
        z = x * x
        z.backward()
        assert_grad_close(x.grad, 6.0)

    def test_diamond_graph(self):
        # y = x + x, z = y * x; dz/dx = 2x² — tests grad accumulation through fan-out
        x = tensor(2.0, requires_grad=True)
        y = x + x
        z = y * x
        z.backward()  # z = y * x = (x + x) * x = 2x * x = 2x^2 ⇒ dz/dx = 4x = 8
        assert_grad_close(x.grad, 8.0)


# ── numerical gradient checks ─────────────────────────────────────────────────


class TestNumericalGradients:
    """Compare analytical gradients against finite differences."""

    def _check(self, f, x_val, eps=1e-5):
        x = tensor(float(x_val), requires_grad=True)
        y = f(x)
        y.backward()
        analytical = float(x.grad.data) if x.grad is not None else zeros_like(x)
        numerical = float(
            numerical_grad(lambda v: f(tensor(float(v))).data, x_val, eps)
        )
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-6)

    def test_add_numerical(self):
        self._check(lambda x: x + tensor(3.0), 2.0)

    def test_mul_numerical(self):
        self._check(lambda x: x * tensor(5.0), 2.0)

    def test_sub_numerical(self):
        self._check(lambda x: x - tensor(1.0), 4.0)

    def test_chain_numerical(self):
        self._check(lambda x: (x * tensor(2.0)) + (x * tensor(3.0)), 1.5)

    def test_quadratic_numerical(self):
        self._check(lambda x: x * x, 3.0)


# ── backward cleanup ──────────────────────────────────────────────────────────


class TestBackwardCleanup:
    def test_children_cleared_after_backward(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=True)
        z = x * y
        z.backward()
        assert z._children == []

    def test_grad_fn_cleared_after_backward(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0)
        z = x + y
        z.backward()
        assert z.grad_fn is None

    def test_keep_graph_preserves_children(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0)
        z = x + y
        z.backward(keep_graph=True)
        assert len(z._children) > 0

    def test_grad_accumulates_across_backward_calls_with_keep_graph(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0)
        z = x + y
        z.backward(keep_graph=True)
        first_grad = float(x.grad.data) if x.grad is not None else zeros_like(x)
        x.grad = None
        z.backward(keep_graph=True)
        second_grad = float(x.grad.data) if x.grad is not None else zeros_like(x)
        assert_grad_close(first_grad, second_grad)
