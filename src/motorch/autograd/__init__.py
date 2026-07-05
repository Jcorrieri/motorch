"""
Autograd stuff
"""

from .forward import apply_forward_pass
from .ufuncs import resolve_local_grads

__all__ = ["apply_forward_pass", "resolve_local_grads"]
