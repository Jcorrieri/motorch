import pytest
import numpy as np
from motorch import Tensor, tensor

# --- Construction --- #

class TestConstruction:
    def test_from_list(self):
        t = Tensor([1, 2, 3])
        assert isinstance(t, Tensor)
        np.testing.assert_array_equal(t.data, np.array([1, 2, 3]))

    def test_from_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        t = Tensor(arr)
        np.testing.assert_array_equal(t.data, arr)

    def test_always_copies(self):
        arr = np.array([1, 2, 3])
        t = Tensor(arr)
        arr[0] = 99
        assert t.data[0] != 99  # should not share memory

    def test_factory_fn(self):
        t = tensor([1, 2, 3], dtype=np.float32)
        assert t.dtype == np.float32

    def test_requires_grad_default(self):
        t = Tensor([1, 2, 3])
        assert t.requires_grad == False

    def test_requires_grad(self):
        t = Tensor([1, 2, 3], requires_grad=True)
        assert t.requires_grad == True


# --- Properties --- #

class TestProperties:
    def test_shape(self):
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)

    def test_dtype(self):
        t = Tensor(np.array([1.0], dtype=np.float64))
        assert t.dtype == np.float64

    def test_ndim(self):
        t = Tensor([[1, 2], [3, 4]])
        assert t.ndim == 2

    def test_T(self):
        t = Tensor([[1, 2], [3, 4]])
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(t.T.data, expected)

    def test_T_is_tensor(self):
        t = Tensor([[1, 2], [3, 4]])
        assert isinstance(t.T, Tensor)

    def test_T_updates_with_data(self):
        t = Tensor([[1, 2], [3, 4]])
        t.data[0, 0] = 99
        assert t.T.data[0, 0] == 99  # recomputed on access


# --- Slicing --- #

class TestSlicing:
    def test_getitem_index(self):
        t = Tensor([1, 2, 3])
        assert t[0].data == 1

    def test_getitem_slice(self):
        t = Tensor([1, 2, 3, 4])
        np.testing.assert_array_equal(t[1:3].data, np.array([2, 3]))

    def test_getitem_returns_tensor(self):
        t = Tensor([1, 2, 3])
        assert isinstance(t[0:2], Tensor)

    def test_getitem_2d(self):
        t = Tensor([[1, 2], [3, 4]])
        np.testing.assert_array_equal(t[0].data, np.array([1, 2]))

    def test_getitem_boolean(self):
        t = Tensor([1, 2, 3, 4])
        np.testing.assert_array_equal(t[t.data > 2].data, np.array([3, 4]))

    def test_setitem(self):
        t = Tensor([1, 2, 3])
        t[0] = 99
        assert t.data[0] == 99

    def test_setitem_tensor(self):
        t = Tensor([1, 2, 3])
        t[0:2] = Tensor([7, 8])
        np.testing.assert_array_equal(t.data[:2], np.array([7, 8]))


# --- Iteration --- #

class TestIteration:
    def test_iter(self):
        t = Tensor([1, 2, 3])
        items = list(t)
        assert len(items) == 3

    def test_iter_yields_tensors(self):
        t = Tensor([1, 2, 3])
        for item in t:
            assert isinstance(item, Tensor)

    def test_len(self):
        t = Tensor([1, 2, 3])
        assert len(t) == 3


# --- Arithmetic (ufuncs) --- #

class TestArithmetic:
    def test_add(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        np.testing.assert_array_equal((a + b).data, np.array([5, 7, 9]))

    def test_mul(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        np.testing.assert_array_equal((a * b).data, np.array([4, 10, 18]))

    def test_scalar_add(self):
        a = Tensor([1, 2, 3])
        np.testing.assert_array_equal((a + 1).data, np.array([2, 3, 4]))

    def test_sub(self):
        a = Tensor([5, 6, 7])
        b = Tensor([1, 2, 3])
        np.testing.assert_array_equal((a - b).data, np.array([4, 4, 4]))

    def test_div(self):
        a = Tensor([4.0, 6.0])
        b = Tensor([2.0, 3.0])
        np.testing.assert_array_equal((a / b).data, np.array([2.0, 2.0]))

    def test_result_is_tensor(self):
        a = Tensor([1, 2, 3])
        b = Tensor([1, 2, 3])
        assert isinstance(a + b, Tensor)


# --- np.* dispatch --- #

class TestNumpyDispatch:
    def test_np_sum(self):
        t = Tensor([1, 2, 3])
        assert np.sum(t) == 6

    def test_np_sum_axis(self):
        t = Tensor([[1, 2], [3, 4]])
        np.testing.assert_array_equal(np.sum(t, axis=0).data, np.array([4, 6]))

    def test_np_sum_returns_scalar(self):
        t = Tensor([1, 2, 3])
        result = np.sum(t)
        assert isinstance(result, (int, float, np.integer, np.floating))

    def test_np_mean(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert np.mean(t) == 2.0

    def test_np_mean_axis(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(np.mean(t, axis=0).data, np.array([2.0, 3.0]))

    def test_np_exp(self):
        t = Tensor([0.0])
        np.testing.assert_allclose(np.exp(t).data, np.array([1.0]))

    def test_np_log1p(self):
        t = Tensor([0.0])
        np.testing.assert_allclose(np.log1p(t).data, np.array([0.0]))

    def test_np_transpose(self):
        t = Tensor([[1, 2], [3, 4]])
        result = np.transpose(t)
        np.testing.assert_array_equal(result.data, np.array([[1, 3], [2, 4]]))

    def test_np_stack(self):
        a = Tensor([1, 2])
        b = Tensor([3, 4])
        result = np.stack([a, b])
        assert isinstance(result, Tensor)
        np.testing.assert_array_equal(result.data, np.array([[1, 2], [3, 4]]))

    def test_np_concatenate(self):
        a = Tensor([1, 2])
        b = Tensor([3, 4])
        result = np.concatenate([a, b])
        np.testing.assert_array_equal(result.data, np.array([1, 2, 3, 4]))

    def test_np_ones(self):
        t = np.ones((2, 3))
        assert isinstance(np.ones((2, 3)), np.ndarray)  # not dispatched
        result = np.ones_like(Tensor(t))
        assert isinstance(result, Tensor)

    def test_np_zeros_like(self):
        t = Tensor([[1, 2], [3, 4]])
        result = np.zeros_like(t)
        np.testing.assert_array_equal(result.data, np.zeros((2, 2)))

    def test_np_where(self):
        t = Tensor([1, -1, 2, -2])
        result = np.where(t.data > 0, t.data, 0)
        np.testing.assert_array_equal(result, np.array([1, 0, 2, 0]))

    def test_np_clip(self):
        t = Tensor([1, 2, 3, 4, 5])
        result = np.clip(t, 2, 4)
        np.testing.assert_array_equal(result.data, np.array([2, 2, 3, 4, 4]))


# --- Instance methods --- #

class TestMethods:
    def test_sum_method(self):
        t = Tensor([1, 2, 3])
        assert t.sum() == 6

    def test_mean_method(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.mean() == 2.0

    def test_transpose_method(self):
        t = Tensor([[1, 2], [3, 4]])
        np.testing.assert_array_equal(t.transpose().data, t.T.data)

    def test_item(self):
        t = Tensor([42])
        assert t.item() == 42

    def test_asarray(self):
        t = Tensor([1, 2, 3])
        arr = np.asarray(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array([1, 2, 3]))
