"""Loss modules for MoTorch."""

from motorch.nn.modules import Module
import motorch.nn.functional as F


class LogisticLoss(Module):
    """Logistic loss module for binary classification with labels in +1/-1."""

    def forward(self, logits, labels):
        """Compute the logistic loss and cache its gradient."""
        self.grad = self._grad(logits, labels)
        return F.logloss(labels, logits)

    def _grad(self, logits, labels):
        return F.logloss_grad(logits, labels)

