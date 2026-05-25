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

    def test_requires_grad(self):
        t = Tensor([1, 2, 3], requires_grad=False)
        assert t.requires_grad == False


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


# --- Instance methods --- #

class TestMethods:
    def test_sum_method(self):
        t = Tensor([1, 2, 3])
        assert t.sum() == 6

    def test_sum_method_axis(self):
        t = Tensor([[1, 2], [3, 4]])
        np.testing.assert_array_equal(t.sum(axis=0).data, np.array([4, 6]))

    def test_mean_method(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.mean() == 2.0

    def test_mean_method_axis(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(t.mean(axis=0).data, np.array([2.0, 3.0]))

    def test_transpose_method(self):
        t = Tensor([[1, 2], [3, 4]])
        np.testing.assert_array_equal(t.transpose().data, t.T.data)

    def test_transpose_method_with_axes(self):
        t = Tensor(np.ones((2, 3, 4)))
        result = t.transpose((2, 0, 1))
        assert result.shape == (4, 2, 3)

    def test_item(self):
        t = Tensor([42])
        assert t.item() == 42

    def test_asarray(self):
        t = Tensor([1, 2, 3])
        arr = np.asarray(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array([1, 2, 3]))

    def test_exp(self):
        t = Tensor([0.0])
        np.testing.assert_allclose(t.exp().data, np.array([1.0]))

    def test_log1p(self):
        t = Tensor([0.0])
        np.testing.assert_allclose(t.log1p().data, np.array([0.0]))

    def test_clip_basic(self):
        t = Tensor([1, 2, 3, 4, 5])
        result = t.clip(2, 4)
        assert isinstance(result, Tensor)
        np.testing.assert_array_equal(result.data, np.array([2, 2, 3, 4, 4]))

    def test_clip_min_only(self):
        t = Tensor([-1, 0, 1, 2])
        result = t.clip(0, None)
        np.testing.assert_array_equal(result.data, np.array([0, 0, 1, 2]))

    def test_clip_max_only(self):
        t = Tensor([1, 2, 3, 4])
        result = t.clip(None, 3)
        np.testing.assert_array_equal(result.data, np.array([1, 2, 3, 3]))

    def test_clip_float(self):
        t = Tensor([0.5, 1.5, 2.5])
        result = t.clip(1.0, 2.0)
        np.testing.assert_allclose(result.data, np.array([1.0, 1.5, 2.0]))

    def test_clip_no_effect(self):
        t = Tensor([2, 3, 4])
        result = t.clip(1, 5)
        np.testing.assert_array_equal(result.data, t.data)


# --- Transpose --- #

class TestTranspose:
    def test_transpose_2d(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.transpose()
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.data, t.data.T)

    def test_transpose_returns_tensor(self):
        t = Tensor([[1, 2], [3, 4]])
        assert isinstance(t.transpose(), Tensor)

    def test_transpose_with_axes(self):
        t = Tensor(np.ones((2, 3, 4)))
        result = t.transpose((2, 0, 1))
        assert result.shape == (4, 2, 3)

    def test_T_vs_transpose_consistent(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(t.T.data, t.transpose().data)

    def test_transpose_1d_noop(self):
        t = Tensor([1, 2, 3])
        np.testing.assert_array_equal(t.transpose().data, t.data)


# --- In-place operations --- #

class TestInPlace:
    def test_iadd_scalar(self):
        t = Tensor([1.0, 2.0, 3.0])
        t += 1
        np.testing.assert_array_equal(t.data, np.array([2.0, 3.0, 4.0]))

    def test_isub_scalar(self):
        t = Tensor([1.0, 2.0, 3.0])
        t -= 1
        np.testing.assert_array_equal(t.data, np.array([0.0, 1.0, 2.0]))

    def test_imul_scalar(self):
        t = Tensor([1.0, 2.0, 3.0])
        t *= 2
        np.testing.assert_array_equal(t.data, np.array([2.0, 4.0, 6.0]))

    def test_idiv_scalar(self):
        t = Tensor([2.0, 4.0, 6.0])
        t /= 2
        np.testing.assert_array_equal(t.data, np.array([1.0, 2.0, 3.0]))

    def test_iadd_tensor(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        a += b
        np.testing.assert_array_equal(a.data, np.array([5.0, 7.0, 9.0]))

    def test_isub_tensor(self):
        a = Tensor([5.0, 6.0, 7.0])
        b = Tensor([1.0, 2.0, 3.0])
        a -= b
        np.testing.assert_array_equal(a.data, np.array([4.0, 4.0, 4.0]))

    def test_inplace_preserves_identity(self):
        """In-place ops should mutate the existing tensor, not create a new one"""
        t = Tensor([1.0, 2.0, 3.0])
        original_id = id(t)
        t += 1
        assert id(t) == original_id


# --- Broadcasting --- #

class TestBroadcasting:
    def test_add_scalar(self):
        t = Tensor([1.0, 2.0, 3.0])
        np.testing.assert_array_equal((t + 5).data, np.array([6.0, 7.0, 8.0]))

    def test_scalar_add_reversed(self):
        t = Tensor([1.0, 2.0, 3.0])
        np.testing.assert_array_equal((5 + t).data, np.array([6.0, 7.0, 8.0]))

    def test_mul_scalar(self):
        t = Tensor([1.0, 2.0, 3.0])
        np.testing.assert_array_equal((t * 3).data, np.array([3.0, 6.0, 9.0]))

    def test_sub_scalar(self):
        t = Tensor([5.0, 6.0, 7.0])
        np.testing.assert_array_equal((t - 2).data, np.array([3.0, 4.0, 5.0]))

    def test_div_scalar(self):
        t = Tensor([4.0, 6.0, 8.0])
        np.testing.assert_array_equal((t / 2).data, np.array([2.0, 3.0, 4.0]))

    def test_broadcast_2d_1d(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([10.0, 20.0])
        np.testing.assert_array_equal(
            (a + b).data, np.array([[11.0, 22.0], [13.0, 24.0]])
        )

    def test_broadcast_result_is_tensor(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert isinstance(t + 1, Tensor)
        assert isinstance(1 + t, Tensor)


# --- Reshape --- #
 
class TestReshape:
    def test_reshape_method_basic(self):
        t = Tensor([1, 2, 3, 4, 5, 6])
        result = t.reshape((2, 3))
        assert result.shape == (2, 3)
 
    def test_reshape_method_returns_tensor(self):
        t = Tensor([1, 2, 3, 4])
        assert isinstance(t.reshape((2, 2)), Tensor)
 
    def test_reshape_method_preserves_data(self):
        t = Tensor([1, 2, 3, 4, 5, 6])
        result = t.reshape((2, 3))
        np.testing.assert_array_equal(result.data, np.array([[1, 2, 3], [4, 5, 6]]))
 
    def test_reshape_method_to_1d(self):
        t = Tensor([[1, 2], [3, 4]])
        result = t.reshape((4,))
        np.testing.assert_array_equal(result.data, np.array([1, 2, 3, 4]))
 
    def test_reshape_method_inferred_dim(self):
        t = Tensor([1, 2, 3, 4, 5, 6])
        result = t.reshape((2, -1))
        assert result.shape == (2, 3)
 
    def test_reshape_method_3d(self):
        t = Tensor(np.arange(24))
        result = t.reshape((2, 3, 4))
        assert result.shape == (2, 3, 4)
 
    def test_reshape_method_same_shape(self):
        t = Tensor([[1, 2], [3, 4]])
        result = t.reshape((2, 2))
        np.testing.assert_array_equal(result.data, t.data)
 
    def test_reshape_method_preserves_dtype(self):
        t = Tensor(np.array([1, 2, 3, 4], dtype=np.float32))
        result = t.reshape((2, 2))
        assert result.dtype == np.float32
 
    def test_module_reshape_basic(self):
        import motorch
        t = Tensor([1, 2, 3, 4, 5, 6])
        result = motorch.reshape(t, (2, 3))
        assert result.shape == (2, 3)
 
    def test_module_reshape_returns_tensor(self):
        import motorch
        t = Tensor([1, 2, 3, 4])
        assert isinstance(motorch.reshape(t, (2, 2)), Tensor)
 
    def test_module_reshape_preserves_data(self):
        import motorch
        t = Tensor([1, 2, 3, 4, 5, 6])
        result = motorch.reshape(t, (2, 3))
        np.testing.assert_array_equal(result.data, np.array([[1, 2, 3], [4, 5, 6]]))
 
    def test_module_reshape_inferred_dim(self):
        import motorch
        t = Tensor([1, 2, 3, 4, 5, 6])
        result = motorch.reshape(t, (3, -1))
        assert result.shape == (3, 2)
 
    def test_module_reshape_3d(self):
        import motorch
        t = Tensor(np.arange(24))
        result = motorch.reshape(t, (2, 3, 4))
        assert result.shape == (2, 3, 4)
 
    def test_method_and_module_consistent(self):
        import motorch
        t = Tensor([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(
            t.reshape((2, 3)).data,
            motorch.reshape(t, (2, 3)).data,
        )


# --- .numpy() method --- #

class TestNumpy:
    def test_numpy_returns_ndarray(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert isinstance(t.numpy(), np.ndarray)

    def test_numpy_values_correct(self):
        t = Tensor([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(t.numpy(), np.array([1.0, 2.0, 3.0]))

    def test_numpy_preserves_dtype(self):
        t = Tensor(np.array([1.0], dtype=np.float32))
        assert t.numpy().dtype == np.float32

    def test_numpy_2d(self):
        t = Tensor([[1, 2], [3, 4]])
        arr = t.numpy()
        np.testing.assert_array_equal(arr, np.array([[1, 2], [3, 4]]))


# --- DAG node creation --- #

class TestNodeCreation:
    def test_children_exist_when_left_requires_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=False)
        result = a + b
        assert result._children is not None
        assert len(result._children) > 0

    def test_children_exist_when_right_requires_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=False)
        b = Tensor([3.0, 4.0], requires_grad=True)
        result = a + b
        assert result._children is not None
        assert len(result._children) > 0

    def test_children_exist_when_both_require_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        result = a + b
        assert result._children is not None
        assert len(result._children) > 0

    def test_no_children_when_no_inputs_require_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=False)
        b = Tensor([3.0, 4.0], requires_grad=False)
        result = a + b
        assert not hasattr(result, '_children') or result._children is None

    def test_children_contain_inputs(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=False)
        result = a + b
        assert a in result._children
        assert b in result._children

    def test_scalar_input_wrapped_as_tensor_in_children(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        result = a + 2.0
        assert all(isinstance(c, Tensor) for c in result._children)

    def test_version_increments_on_inplace(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        version_before = a._version
        a += 1.0
        assert a._version == version_before + 1

    def test_version_unchanged_on_out_of_place(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        version_before = a._version
        _ = a + 1.0
        assert a._version == version_before

    def test_version_increments_each_inplace(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        version_before = a._version
        a += 1.0
        a += 1.0
        a += 1.0
        assert a._version == version_before + 3

    def test_inplace_no_children_when_no_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=False)
        a += 1.0
        assert not hasattr(a, '_children') or a._children is None

    def test_version_not_incremented_on_inplace_no_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=False)
        version_before = a._version
        a += 1.0
        assert a._version == version_before


# --- Gradient computation --- #

class TestGradients:
    # -- Addition -- #

    def test_add_grad_left(self):
        x = tensor(2.0)
        y = tensor(3.0)
        z = x + y
        z.grad = 1.0
        assert z._children[0].grad_fn() == 1.0  # dz/dx = 1

    def test_add_grad_right(self):
        x = tensor(2.0)
        y = tensor(3.0)
        z = x + y
        z.grad = 1.0
        assert z._children[1].grad_fn() == 1.0  # dz/dy = 1

    def test_add_grad_scales_with_upstream(self):
        x = tensor(2.0)
        y = tensor(3.0)
        z = x + y
        z.grad = 5.0
        assert z._children[0].grad_fn() == 5.0  # dz/dx * upstream = 1 * 5
        assert z._children[1].grad_fn() == 5.0  # dz/dy * upstream = 1 * 5

    # -- Multiplication -- #

    def test_mul_grad_left(self):
        x = tensor(2.0)
        y = tensor(5.0)
        z = x * y
        z.grad = 1.0
        assert z._children[0].grad_fn() == 5.0  # dz/dx = y = 5

    def test_mul_grad_right(self):
        x = tensor(2.0)
        y = tensor(5.0)
        z = x * y
        z.grad = 1.0
        assert z._children[1].grad_fn() == 2.0  # dz/dy = x = 2

    def test_mul_grad_scales_with_upstream(self):
        x = tensor(2.0)
        y = tensor(5.0)
        z = x * y
        z.grad = 3.0
        assert z._children[0].grad_fn() == 15.0  # dz/dx * upstream = 5 * 3
        assert z._children[1].grad_fn() == 6.0   # dz/dy * upstream = 2 * 3

    # -- Chained addition -- #

    def test_add_chain(self):
        # z = x + y, w = z + v
        # dw/dx = 1, dw/dy = 1, dw/dv = 1
        x = tensor(1.0)
        y = tensor(2.0)
        v = tensor(3.0)
        z = x + y
        w = z + v
        w.grad = 1.0
        z.grad = w._children[0].grad_fn()  # propagate to z
        assert z.grad == 1.0
        assert z._children[0].grad_fn() == 1.0  # dw/dx
        assert z._children[1].grad_fn() == 1.0  # dw/dy

    # -- Chained multiplication -- #

    def test_mul_chain(self):
        # z = x * y, w = z * v
        # dw/dz = v, dw/dx = y * v, dw/dy = x * v
        x = tensor(2.0)
        y = tensor(3.0)
        v = tensor(4.0)
        z = x * y
        w = z * v
        w.grad = 1.0
        z.grad = w._children[0].grad_fn()  # dw/dz = v = 4
        assert z.grad == 4.0
        assert z._children[0].grad_fn() == 12.0  # dw/dx = y * v = 3 * 4
        assert z._children[1].grad_fn() == 8.0   # dw/dy = x * v = 2 * 4

    # -- Mixed addition and multiplication -- #

    def test_mul_then_add(self):
        # z = x * y, w = z + v
        # dw/dz = 1, dw/dx = y, dw/dy = x, dw/dv = 1
        x = tensor(2.0)
        y = tensor(3.0)
        v = tensor(4.0)
        z = x * y
        w = z + v
        w.grad = 1.0
        z.grad = w._children[0].grad_fn()  # dw/dz = 1
        assert z.grad == 1.0
        assert z._children[0].grad_fn() == 3.0  # dw/dx = y = 3
        assert z._children[1].grad_fn() == 2.0  # dw/dy = x = 2
        assert w._children[1].grad_fn() == 1.0  # dw/dv = 1

    def test_add_then_mul(self):
        # z = x + y, w = z * v
        # dw/dz = v, dw/dx = v, dw/dy = v, dw/dv = z = x + y
        x = tensor(2.0)
        y = tensor(3.0)
        v = tensor(4.0)
        z = x + y
        w = z * v
        w.grad = 1.0
        z.grad = w._children[0].grad_fn()  # dw/dz = v = 4
        assert z.grad == 4.0
        assert z._children[0].grad_fn() == 4.0  # dw/dx = v = 4
        assert z._children[1].grad_fn() == 4.0  # dw/dy = v = 4
        assert w._children[1].grad_fn() == 5.0  # dw/dv = z = x + y = 5

    # -- grad_fn return type -- #

    def test_grad_fn_returns_numeric(self):
        x = tensor(2.0)
        y = tensor(3.0)
        z = x * y
        z.grad = 1.0
        result = z._children[0].grad_fn()
        assert isinstance(result, (int, float, np.integer, np.floating))
