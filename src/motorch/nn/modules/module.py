"""
Ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L407
"""
from typing import Any, Union

from ..parameter import Parameter
from motorch import Tensor


class Module():
    def __init__(self):
        # Bypass Module.__setattr__() which has special use
        super().__setattr__("training", True)
        super().__setattr__("_parameters", {})
        super().__setattr__("_modules", {})

    # For distinguishing between trainable parameters and other attributes
    def __setattr__(self, name: str, value: Any, /) -> None:
        if isinstance(value, Parameter):
            params = self.__dict__.get("_parameters", {})
            params[name] = value
        elif isinstance(value, Module):
            modules = self.__dict__.get("_modules", {})
            modules[name] = value 
        else:
            object.__setattr__(self, name, value)

    # Needed to support __getattr__ override
    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
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

    def __str__(self):
        string = "Model:\n-----------\n"

        for module, value in getattr(self, "_modules").items():
            string += f"{module}: {str(value)}"

        return string

    def parameters(self, recurse=True):
        for param in getattr(self, "_parameters").values():
            yield param
        if recurse:
            for module in getattr(self, "_modules").values():
                yield from module.parameters()

    def train(self, mode=True):
        self.training = mode
        for module in getattr(self, "_modules").values():
            module.train(mode)   # propagates to all children
        return self

    def eval(self):
        return self.train(False)

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError
    
    # TODO: Implement
    def zero_grads(self):
        pass

    def backwards(self):
        pass
