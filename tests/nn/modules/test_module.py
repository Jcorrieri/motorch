import pytest
import numpy as np
from motorch import Tensor
from motorch.nn import Parameter
from motorch.nn.modules import Module


# --- Concrete subclasses for testing --- #

class SimpleModule(Module):
    """Minimal module with a forward pass"""
    def forward(self, x):
        return x


class LinearModule(Module):
    """Module with parameters, simulating a linear layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(Tensor(np.random.randn(out_features, in_features)))
        self.bias = Parameter(Tensor(np.zeros(out_features)))

    def forward(self, x):
        return x @ self.weight.data.T + self.bias.data


class NestedModule(Module):
    """Module with child submodules"""
    def __init__(self):
        super().__init__()
        self.layer1 = LinearModule(4, 8)
        self.layer2 = LinearModule(8, 1)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class ActivationModule(Module):
    """Parameter-free module"""
    def forward(self, x):
        return Tensor(np.maximum(x.data, 0))


# --- Initialization --- #

class TestInit:
    def test_training_default_true(self):
        m = SimpleModule()
        assert m.training == True

    def test_parameters_empty(self):
        m = SimpleModule()
        assert m._parameters == {}

    def test_modules_empty(self):
        m = SimpleModule()
        assert m._modules == {}

    def test_init_does_not_call_setattr(self):
        """training/_parameters/_modules should be set via super().__setattr__"""
        m = SimpleModule()
        assert "training" in m.__dict__
        assert "_parameters" in m.__dict__
        assert "_modules" in m.__dict__


# --- __setattr__ --- #

class TestSetAttr:
    def test_parameter_goes_to_parameters_dict(self):
        m = SimpleModule()
        p = Parameter(Tensor([1.0, 2.0]))
        m.weight = p
        assert "weight" in m._parameters # type: ignore[reportOperatorIssue]
        assert m._parameters["weight"] is p # type: ignore[reportIndexIssue]

    def test_module_goes_to_modules_dict(self):
        m = SimpleModule()
        child = SimpleModule()
        m.child = child
        assert "child" in m._modules # type: ignore[reportOperatorIssue]
        assert m._modules["child"] is child # type: ignore[reportIndexIssue]

    def test_plain_attr_goes_to_instance_dict(self):
        m = SimpleModule()
        m.learning_rate = 0.01
        assert m.__dict__["learning_rate"] == 0.01
        assert "learning_rate" not in m._parameters # type: ignore[reportOperatorIssue]
        assert "learning_rate" not in m._modules # type: ignore[reportOperatorIssue]

    def test_tensor_goes_to_instance_dict(self):
        """Plain Tensors (non-Parameter) should not go to _parameters"""
        m = SimpleModule()
        m.cache = Tensor([1.0, 2.0])
        assert "cache" not in m._parameters # type: ignore[reportOperatorIssue]
        assert "cache" in m.__dict__

    def test_parameter_not_in_instance_dict(self):
        m = SimpleModule()
        m.weight = Parameter(Tensor([1.0]))
        assert "weight" not in m.__dict__

    def test_module_not_in_instance_dict(self):
        m = SimpleModule()
        m.child = SimpleModule()
        assert "child" not in m.__dict__


# --- __getattr__ --- #

class TestGetAttr:
    def test_get_parameter(self):
        m = LinearModule(2, 3)
        assert isinstance(m.weight, Parameter)

    def test_get_module(self):
        m = NestedModule()
        assert isinstance(m.layer1, LinearModule)

    def test_get_missing_raises(self):
        m = SimpleModule()
        with pytest.raises(AttributeError):
            _ = m.nonexistent

    def test_getattr_error_message(self):
        m = SimpleModule()
        with pytest.raises(AttributeError, match="SimpleModule"):
            _ = m.nonexistent

    def test_plain_attr_accessible(self):
        m = SimpleModule()
        m.lr = 0.1
        assert m.lr == 0.1


# --- parameters() --- #

class TestParameters:
    def test_yields_own_parameters(self):
        m = LinearModule(2, 3)
        params = list(m.parameters())
        assert len(params) == 2  # weight and bias

    def test_yields_parameter_instances(self):
        m = LinearModule(2, 3)
        for p in m.parameters():
            assert isinstance(p, Parameter)

    def test_no_parameters_empty(self):
        m = ActivationModule()
        assert list(m.parameters()) == []

    def test_recursive_parameters(self):
        m = NestedModule()
        params = list(m.parameters())
        assert len(params) == 4  # layer1.weight, layer1.bias, layer2.weight, layer2.bias

    def test_non_recursive_parameters(self):
        m = NestedModule()
        params = list(m.parameters(recurse=False))
        assert len(params) == 0  # NestedModule has no direct parameters

    def test_parameters_is_generator(self):
        m = LinearModule(2, 3)
        import types
        assert isinstance(m.parameters(), types.GeneratorType)

    def test_parameters_correct_shapes(self):
        m = LinearModule(2, 3)
        params = list(m.parameters())
        shapes = {p.shape for p in params}
        assert (3, 2) in shapes  # weight shape
        assert (3,) in shapes    # bias shape


# --- train() / eval() --- #

class TestTrainEval:
    def test_default_training_true(self):
        m = SimpleModule()
        assert m.training == True

    def test_eval_sets_false(self):
        m = SimpleModule()
        m.eval()
        assert m.training == False

    def test_train_sets_true(self):
        m = SimpleModule()
        m.eval()
        m.train()
        assert m.training == True

    def test_train_returns_self(self):
        m = SimpleModule()
        result = m.train()
        assert result is m

    def test_eval_returns_self(self):
        m = SimpleModule()
        result = m.eval()
        assert result is m

    def test_train_propagates_to_children(self):
        m = NestedModule()
        m.eval()
        assert m.layer1.training == False
        assert m.layer2.training == False
        m.train()
        assert m.layer1.training == True
        assert m.layer2.training == True

    def test_eval_propagates_to_children(self):
        m = NestedModule()
        m.eval()
        assert m.training == False
        assert m.layer1.training == False
        assert m.layer2.training == False

    def test_train_mode_false(self):
        m = SimpleModule()
        m.train(False)
        assert m.training == False


# --- forward() / __call__ --- #

class TestForward:
    def test_forward_not_implemented(self):
        m = Module()
        with pytest.raises(NotImplementedError):
            m.forward(Tensor([1.0]))

    def test_call_invokes_forward(self):
        m = SimpleModule()
        x = Tensor([1.0, 2.0])
        result = m(x)
        np.testing.assert_array_equal(result.data, x.data)

    def test_linear_forward_shape(self):
        m = LinearModule(4, 3)
        x = Tensor(np.ones(4))
        result = m(x)
        assert result.shape == (3,)

    def test_nested_forward(self):
        m = NestedModule()
        x = Tensor(np.ones(4))
        result = m(x)
        assert result.shape == (1,)

    def test_activation_forward(self):
        m = ActivationModule()
        x = Tensor([-1.0, 0.0, 1.0, 2.0])
        result = m(x)
        np.testing.assert_array_equal(result.data, np.array([0.0, 0.0, 1.0, 2.0]))


# --- __str__ --- #

class TestStr:
    def test_str_contains_model(self):
        m = NestedModule()
        assert "Model:" in str(m)

    def test_str_contains_layer_names(self):
        m = NestedModule()
        s = str(m)
        assert "layer1" in s
        assert "layer2" in s

    def test_str_no_modules(self):
        m = SimpleModule()
        s = str(m)
        assert "Model:" in s


# --- Composition --- #

class TestComposition:
    def test_nested_modules_registered(self):
        m = NestedModule()
        assert "layer1" in m._modules # type: ignore[reportOperatorIssue]
        assert "layer2" in m._modules # type: ignore[reportOperatorIssue]

    def test_nested_module_accessible(self):
        m = NestedModule()
        assert isinstance(m.layer1, LinearModule)
        assert isinstance(m.layer2, LinearModule)

    def test_parameter_free_child(self):
        class NetWithActivation(Module):
            def __init__(self):
                super().__init__()
                self.linear = LinearModule(2, 2)
                self.relu = ActivationModule()
            def forward(self, x):
                return self.relu(self.linear(x))

        m = NetWithActivation()
        params = list(m.parameters())
        assert len(params) == 2  # only linear's weight and bias

    def test_deeply_nested_parameters(self):
        class Deep(Module):
            def __init__(self):
                super().__init__()
                self.a = NestedModule()  # contains 4 params
                self.b = LinearModule(2, 2)  # contains 2 params
            def forward(self, x): pass

        m = Deep()
        assert len(list(m.parameters())) == 6
