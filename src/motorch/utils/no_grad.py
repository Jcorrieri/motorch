"""
Serves the same purpose as torch's no_grad() context manager
"""

class no_grad:
    _grad_enabled = True

    def __enter__(self):
        self._prev = no_grad._grad_enabled
        no_grad._grad_enabled = False

    def __exit__(self, *args):
        no_grad._grad_enabled = self._prev


def _requires_grad(inputs):
    return no_grad._grad_enabled and any([t.requires_grad for t in inputs])
