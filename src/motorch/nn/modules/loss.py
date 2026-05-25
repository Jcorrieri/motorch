"""Loss modules for MoTorch."""

from motorch.nn.modules import Module
from motorch.tensor import tensor, tensor_ones_like, tensor_zeros_like
from motorch.utils import no_grad, _track_grads
import motorch.nn.functional as F


class LogisticLoss(Module):
    """Logistic loss module for binary classification with labels in +1/-1."""

    def forward(self, logits, labels):
        """Compute the logistic loss and cache its gradient."""
        with no_grad():
            out = F.logloss(labels, logits)

        track_grads = _track_grads([logits])
        result = tensor(out.data, requires_grad=track_grads)
        if _track_grads([logits]):
            result._children = [logits, labels]
            logits_local_grad = self._grad(logits, labels)
            labels_local_grad = tensor_ones_like(labels).numpy() # TODO: Add label grads
            logits.grad_fn = lambda _g=logits_local_grad: result.grad * _g
            labels.grad_fn = lambda _g=labels_local_grad: result.grad * _g
            result.grad = tensor_zeros_like(result.data).numpy()
            
        return result

    def _grad(self, logits, labels):
        with no_grad():
            out = F.logloss_grad(logits, labels)
        return out

