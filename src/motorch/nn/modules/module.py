"""
Ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L407
"""

from collections.abc import Sequence
from motorch.nn.parameter import Parameter
from motorch import Tensor
from motorch.tensor import tensor_zeros_like
from motorch.utils import format, _track_grads


class Module():
    """Base class for neural network modules.

    Modules can contain both parameters and child modules. They support
    recursive ``train``/``eval`` mode propagation and provide hooks for
    custom forward computation and representation.
    """

    def __init__(self):
        # Bypass Module.__setattr__() which has special use
        super().__setattr__("training", True)
        super().__setattr__("_parameters", {})
        super().__setattr__("_modules", {})

    # For distinguishing between trainable parameters and other attributes
    def __setattr__(self, name: str, value, /) -> None:
        """Register Parameters and child Modules automatically.

        Parameters assigned to module attributes are stored in internal
        dictionaries for later traversal and optimization.
        """
        if isinstance(value, Parameter):
            params = self.__dict__.get("_parameters", {})
            params[name] = value
        elif isinstance(value, Module):
            modules = self.__dict__.get("_modules", {})
            modules[name] = value 
        else:
            object.__setattr__(self, name, value)

    # Needed to support __getattr__ override
    def __getattr__(self, name: str) -> "Tensor | Module":
        """Retrieve registered parameters, buffers, or child modules."""
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_parameters(self, recurse=True):
        """Yield module parameter name, parameter pairs.

        When ``recurse`` is True, this method traverses child modules recursively.
        """
        for name, param in getattr(self, "_parameters").items():
            yield name, param
        if recurse:
            for module in getattr(self, "_modules").values():
                yield from module.named_parameters()

    def parameters(self, recurse=True):
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def train(self, mode=True):
        """Set the module to training mode and propagate to child modules."""
        self.training = mode
        for module in getattr(self, "_modules").values():
            module.train(mode)   # propagates to all children
        return self

    def eval(self):
        """Set the module to evaluation mode."""
        return self.train(False)

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def _backwards(self, result: Tensor, inputs: Sequence, local_grads: Sequence):
        if _track_grads(inputs):
            result._children = list(inputs)
            result.grad = tensor_zeros_like(result).numpy()
            result.requires_grad = True
            for input, grad in zip(inputs, local_grads):
                input.grad_fn = lambda _g=grad: result.grad * (_g.data if isinstance(_g, Tensor) else _g)
        else:
            result.requires_grad = False
    
    # NOTE: I stole most of the following from PyTorch directly
    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Return the extra representation of the module.

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ""

    def __repr__(self) -> str:
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in getattr(self, "_modules").items():
            mod_str = repr(module)
            mod_str = format._addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    
