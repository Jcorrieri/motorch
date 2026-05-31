from motorch.utils.no_grad import no_grad, _requires_grad


def apply_forward_pass(z, inputs, local_grads):
    if not _requires_grad(inputs): 
        return

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
