"""Loss modules for MoTorch."""

from motorch.nn.modules import Module
from motorch.tensor import tensor, tensor_zeros_like 
from motorch.utils import no_grad, _requires_grad
import motorch.nn.functional as F


class LogisticLoss(Module):
    """Logistic loss module for binary classification with labels in +1/-1."""

    def forward(self, logits, labels):
        """Compute the logistic loss and cache its gradient."""
        with no_grad():
            out = F.logloss(labels, logits) 

        requires_grad = _requires_grad([logits])
        result = tensor(out.data, requires_grad=requires_grad)
        if requires_grad:
            result.grad_fn = lambda _z=result: self._grad_fn(_z, logits, labels)
            result._children = [logits]

        return result

    def _grad_fn(self, z, x, y):
        with no_grad():
            local_grad = F.logloss_grad(x, y)
            for item in [x, y]:
                if not item.grad:
                    item.grad = tensor_zeros_like(item)
            x.grad += z.grad * local_grad

