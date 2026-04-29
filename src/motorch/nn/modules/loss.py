import motorch as mo
from motorch.nn.functional import sigmoid


# Logistic Loss for +1/-1 labels, averaged over batch
class LogisticLoss():
    def __call__(self, logits, labels):
        self.grad = self._grad(logits, labels)
        return mo.mean(mo.log1p(mo.exp(-labels * logits)))

    def __str__(self):
        return "Logistic Loss Function"

    def _grad(self, logits, labels):
        return mo.tensor(-labels * sigmoid(-labels * logits) / len(labels))

