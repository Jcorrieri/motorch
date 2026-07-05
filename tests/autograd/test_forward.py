"""Tests for motorch apply_forward_pass."""

import numpy as np
import pytest

from motorch.tensor import tensor, Tensor
from motorch.autograd.forward import apply_forward_pass


# ── helpers ───────────────────────────────────────────────────────────────────


def make_output(val=0.0):
    """Create a plain output tensor with no graph attached."""
    return tensor(val)


def assert_grad_close(actual, expected, rtol=1e-5, atol=1e-8):
    np.testing.assert_allclose(
        actual.data if isinstance(actual, Tensor) else np.array(actual),
        expected.data if isinstance(expected, Tensor) else np.array(expected),
        rtol=rtol,
        atol=atol,
    )


# ── input validation ──────────────────────────────────────────────────────────


class TestInputValidation:
    def test_inputs_not_sequence_raises(self):
        z = make_output()
        x = tensor(1.0, requires_grad=True)
        with pytest.raises(AssertionError, match="inputs must be a sequence"):
            apply_forward_pass(z, x, [tensor(1.0)])  # bare tensor, not a list

    def test_local_grads_not_sequence_raises(self):
        z = make_output()
        x = tensor(1.0, requires_grad=True)
        with pytest.raises(AssertionError, match="local_grads must be a sequence"):
            apply_forward_pass(z, [x], tensor(1.0))  # bare tensor, not a list

    def test_mismatched_lengths_raises(self):
        z = make_output()
        x = tensor(1.0, requires_grad=True)
        y = tensor(2.0, requires_grad=True)
        with pytest.raises(AssertionError, match="same length"):
            apply_forward_pass(z, [x, y], [tensor(1.0)])  # 2 inputs, 1 grad

    def test_no_local_grads_and_no_ufunc_raises(self):
        z = make_output()
        x = tensor(1.0, requires_grad=True)
        with pytest.raises(AssertionError, match="no ufunc was passed"):
            apply_forward_pass(z, [x], None)  # neither local_grads nor ufunc

    def test_no_requires_grad_returns_early(self):
        z = make_output()
        x = tensor(1.0, requires_grad=False)
        y = tensor(2.0, requires_grad=False)
        apply_forward_pass(z, [x, y], [tensor(1.0), tensor(1.0)])
        assert z.grad_fn is None
        assert z._children == []
        assert z.requires_grad is False


# ── graph construction ────────────────────────────────────────────────────────


class TestGraphConstruction:
    def test_requires_grad_set_on_output(self):
        z = make_output()
        x = tensor(1.0, requires_grad=True)
        apply_forward_pass(z, [x], [tensor(1.0)])
        assert z.requires_grad is True

    def test_grad_fn_set_on_output(self):
        z = make_output()
        x = tensor(1.0, requires_grad=True)
        apply_forward_pass(z, [x], [tensor(1.0)])
        assert z.grad_fn is not None
        assert callable(z.grad_fn)

    def test_children_set_on_output(self):
        z = make_output()
        x = tensor(1.0, requires_grad=True)
        y = tensor(2.0, requires_grad=False)
        inputs = [x, y]
        apply_forward_pass(z, inputs, [tensor(1.0), tensor(1.0)])
        assert z._children is inputs

    def test_no_grad_input_skipped(self):
        z = make_output()
        x = tensor(1.0, requires_grad=False)
        apply_forward_pass(z, [x], [tensor(1.0)])
        assert z.grad_fn is None

    def test_partial_requires_grad_sets_graph(self):
        # Only one input requires grad — graph should still be built
        z = make_output()
        x = tensor(1.0, requires_grad=True)
        y = tensor(2.0, requires_grad=False)
        apply_forward_pass(z, [x, y], [tensor(1.0), tensor(1.0)])
        assert z.grad_fn is not None
        assert z.requires_grad is True


# ── gradient computation ──────────────────────────────────────────────────────


class TestGradientComputation:
    def test_single_input_grad_initialized(self):
        # x.grad starts as None — should be set to total_grad
        z = make_output(1.0)
        x = tensor(2.0, requires_grad=True)
        local_grad = tensor(3.0)
        apply_forward_pass(z, [x], [local_grad])
        z.grad = tensor(1.0)
        if z.grad_fn:
            z.grad_fn()
        assert_grad_close(x.grad, 3.0)  # 1.0 * 3.0

    def test_single_input_grad_scales_with_upstream(self):
        z = make_output(1.0)
        x = tensor(2.0, requires_grad=True)
        local_grad = tensor(3.0)
        apply_forward_pass(z, [x], [local_grad])
        z.grad = tensor(2.0)
        if z.grad_fn:
            z.grad_fn()
        assert_grad_close(x.grad, 6.0)  # 2.0 * 3.0

    def test_two_inputs_grad_computed_independently(self):
        z = make_output(1.0)
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=True)
        apply_forward_pass(z, [x, y], [tensor(4.0), tensor(5.0)])
        z.grad = tensor(1.0)
        if z.grad_fn:
            z.grad_fn()
        assert_grad_close(x.grad, 4.0)
        assert_grad_close(y.grad, 5.0)

    def test_grad_accumulates_when_already_set(self):
        # x.grad already has a value — should += not overwrite
        z = make_output(1.0)
        x = tensor(2.0, requires_grad=True)
        x.grad = tensor(10.0)
        apply_forward_pass(z, [x], [tensor(3.0)])
        z.grad = tensor(1.0)
        if z.grad_fn:
            z.grad_fn()
        assert_grad_close(x.grad, 13.0)  # 10.0 + 1.0 * 3.0

    def test_grad_none_then_set(self):
        z = make_output(1.0)
        x = tensor(2.0, requires_grad=True)
        assert x.grad is None
        apply_forward_pass(z, [x], [tensor(1.0)])
        z.grad = tensor(1.0)
        if z.grad_fn:
            z.grad_fn()
        assert x.grad is not None

    def test_vector_grad_computation(self):
        z = tensor([1.0, 2.0, 3.0])
        x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        local_grad = tensor([2.0, 2.0, 2.0])
        apply_forward_pass(z, [x], [local_grad])
        z.grad = tensor([1.0, 1.0, 1.0])
        if z.grad_fn:
            z.grad_fn()
        assert_grad_close(x.grad, [2.0, 2.0, 2.0])


# ── in-place mutation detection ───────────────────────────────────────────────


class TestInplaceMutationDetection:
    def test_inplace_mutation_raises(self):
        z = make_output(1.0)
        x = tensor(2.0, requires_grad=True)
        apply_forward_pass(z, [x], [tensor(1.0)])
        z.grad = tensor(1.0)
        x += 1.0  # mutates x, bumps version
        with pytest.raises(ValueError):
            if z.grad_fn:
                z.grad_fn()

    def test_no_mutation_does_not_raise(self):
        z = make_output(1.0)
        x = tensor(2.0, requires_grad=True)
        apply_forward_pass(z, [x], [tensor(1.0)])
        z.grad = tensor(1.0)
        if z.grad_fn:
            z.grad_fn()  # should not raise

    def test_out_of_place_op_does_not_raise(self):
        z = make_output(1.0)
        x = tensor(2.0, requires_grad=True)
        apply_forward_pass(z, [x], [tensor(1.0)])
        z.grad = tensor(1.0)
        _ = x + 1.0  # out-of-place, should not bump version
        if z.grad_fn:
            z.grad_fn()  # should not raise


# ── ufunc path ────────────────────────────────────────────────────────────────


class TestUfuncPath:
    def test_ufunc_overrides_local_grads(self):
        import numpy as np

        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=True)
        z = tensor(5.0)
        apply_forward_pass(z, [x, y], None, ufunc=np.add)
        z.grad = tensor(1.0)
        if z.grad_fn:
            z.grad_fn()
        # d(x+y)/dx = 1, d(x+y)/dy = 1
        assert_grad_close(x.grad, 1.0)
        assert_grad_close(y.grad, 1.0)

    def test_ufunc_mul_grads(self):
        import numpy as np

        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=True)
        z = tensor(6.0)
        apply_forward_pass(z, [x, y], None, ufunc=np.multiply)
        z.grad = tensor(1.0)
        if z.grad_fn:
            z.grad_fn()
        # d(x*y)/dx = y = 3, d(x*y)/dy = x = 2
        assert_grad_close(x.grad, 3.0)
        assert_grad_close(y.grad, 2.0)
