from motorch.nn.modules import Module
import motorch.nn.functional as F


# Logistic Loss for +1/-1 labels, averaged over batch
class LogisticLoss(Module):
    def forward(self, logits, labels):
        self.grad = self._grad(logits, labels)
        return F.logloss(labels, logits)

    def __str__(self):
        return "LogisticLoss()"

    def _grad(self, logits, labels):
        return F.logloss_grad(logits, labels)

