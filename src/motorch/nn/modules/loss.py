"""Loss modules for MoTorch."""

from motorch.autograd.forward import apply_forward_pass
from motorch.nn.modules import Module
from motorch.tensor import tensor 
from motorch.utils import no_grad 
import motorch.nn.functional as F


class LogisticLoss(Module):
    """Logistic loss module for binary classification with labels in +1/-1."""

    def forward(self, logits, labels):
        """Compute the logistic loss and cache its gradient."""
        with no_grad():
            out = F.logloss(labels, logits) 

        result = tensor(out.data)
        local_grads = [self._grad_fn(logits, labels)]
        apply_forward_pass(result, [logits], local_grads)

        return result

    def _grad_fn(self, x, y):
        with no_grad():
            x_local_grad = F.logloss_grad(x, y)
        return x_local_grad # TODO: Return label grads

