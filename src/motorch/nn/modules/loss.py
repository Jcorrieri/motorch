import numpy as np

from motorch.nn.functional import sigmoid


# Logistic Loss for +1/-1 labels, averaged over batch
class LogisticLoss():
    def __call__(self, logits, labels):
        self.grad = self._grad(logits, labels)
        return np.mean(np.log1p(np.exp(-labels * logits)))

    def __str__(self):
        return "Logistic Loss Function"

    def _grad(self, logits, labels):
        return -labels * sigmoid(-labels * logits) / len(labels)
