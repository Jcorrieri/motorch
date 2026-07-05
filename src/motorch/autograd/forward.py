from collections.abc import Sequence
from motorch.utils.no_grad import no_grad, _requires_grad
from .ufuncs import resolve_local_grads


def apply_forward_pass(z, inputs, local_grads, ufunc=None):
    assert isinstance(inputs, Sequence), (
        f"inputs must be a sequence, got {type(inputs)}"
    )
    assert isinstance(local_grads, Sequence) or local_grads is None, (
        f"local_grads must be a sequence, got {type(local_grads)}"
    )

    if local_grads:
        assert len(inputs) == len(local_grads), (
            f"inputs and local_grads must have same length, got {len(inputs)} and {len(local_grads)}"
        )
    else:
        assert ufunc, "local_grads is None but no ufunc was passed"

    if not _requires_grad(inputs):
        return

    if ufunc:
        local_grads = resolve_local_grads(ufunc, inputs)

    def grad_fn():
        expected_version = z._version
        with no_grad():
            for i, x in enumerate(inputs):
                total_grad = z.grad * local_grads[i]
                if x.grad is None:
                    x.grad = total_grad
                else:
                    x.grad += total_grad
                if x._version != expected_version:
                    raise ValueError(f"{x} has been modified in-place.")

    z.requires_grad = True
    z.grad_fn = grad_fn
    z._children = inputs
