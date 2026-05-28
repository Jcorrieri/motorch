"""Loss modules for MoTorch."""

from motorch.nn.modules import Module
from motorch.tensor import tensor, tensor_ones_like
from motorch.utils import no_grad 
import motorch.nn.functional as F


class LogisticLoss(Module):
    """Logistic loss module for binary classification with labels in +1/-1."""

    def forward(self, logits, labels):
        """Compute the logistic loss and cache its gradient."""
        with no_grad():
            out = F.logloss(labels, logits) 

        result = tensor(out.data)
        logits_local_grad = self._grad(logits, labels)
        labels_local_grad = tensor_ones_like(labels).numpy() # TODO: Add label grads
        self._backwards(result, (labels, logits), (labels_local_grad, logits_local_grad))

        return result

    def _grad(self, logits, labels):
        with no_grad():
            out = F.logloss_grad(logits, labels)
        return out

