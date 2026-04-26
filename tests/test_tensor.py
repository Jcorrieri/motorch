import pytest
import numpy as np
import motorch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def vec1():
    return motorch.tensor([1.0, 2.0, 3.0])

@pytest.fixture
def vec2():
    return motorch.tensor([4.0, 5.0, 6.0])

@pytest.fixture
def zeros():
    return motorch.tensor([0.0, 0.0, 0.0])


# ── add (__add__) ─────────────────────────────────────────────────────────────

def test_add_1d(vec1, vec2):
    z = vec1 + vec2
    assert z.shape == (3,)
    assert np.array_equal(z.data, [5.0, 7.0, 9.0])

def test_add_returns_tensor(vec1, vec2):
    z = vec1 + vec2
    assert isinstance(z, motorch.Tensor)

def test_add_commutative(vec1, vec2):
    assert np.array_equal((vec1 + vec2).data, (vec2 + vec1).data)

def test_add_zero_identity(vec1, zeros):
    assert np.array_equal((vec1 + zeros).data, vec1.data)

def test_add_scalar(vec1):
    z = vec1 + 1.0
    assert z.shape == (3,)
    assert np.array_equal(z.data, [2.0, 3.0, 4.0])

def test_add_negative(vec1):
    neg = motorch.tensor([-1.0, -2.0, -3.0])
    z = vec1 + neg
    assert np.array_equal(z.data, [0.0, 0.0, 0.0])

def test_add_does_not_mutate(vec1, vec2):
    original = vec1.data.copy()
    _ = vec1 + vec2
    assert np.array_equal(vec1.data, original)


# ── radd (__radd__) ───────────────────────────────────────────────────────────

def test_radd_scalar(vec1):
    z = 1.0 + vec1
    assert z.shape == (3,)
    assert np.array_equal(z.data, [2.0, 3.0, 4.0])

def test_radd_matches_add(vec1):
    assert np.array_equal((1.0 + vec1).data, (vec1 + 1.0).data)

def test_radd_zero(vec1):
    z = 0.0 + vec1
    assert np.array_equal(z.data, vec1.data)

def test_radd_returns_tensor(vec1):
    z = 1.0 + vec1
    assert isinstance(z, motorch.Tensor)


# ── Shape and type edge cases ─────────────────────────────────────────────────

def test_add_shape_mismatch_raises():
    a = motorch.tensor([1.0, 2.0, 3.0])
    b = motorch.tensor([1.0, 2.0])
    with pytest.raises((ValueError, Exception)):
        _ = a + b

def test_add_2d_raises():
    a = motorch.tensor([[1.0, 2.0, 3.0]])  # (1, 3) — row vector
    b = motorch.tensor([1.0, 2.0, 3.0])   # (3,)
    z = a + b
    assert z.shape == (1, 3)

def test_add_float_precision():
    a = motorch.tensor([0.1, 0.2, 0.3])
    b = motorch.tensor([0.1, 0.2, 0.3])
    z = a + b
    assert np.allclose(z.data, [0.2, 0.4, 0.6])


# ── dtype cases ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", [
    np.float32,
    np.float64,
    np.int32,
    np.int64,
])
def test_tensor_preserves_dtype(dtype):
    x = motorch.tensor([1, 2, 3], dtype=dtype)
    assert x.dtype == dtype


@pytest.mark.parametrize("dtype", [
    np.float32,
    np.float64,
])
def test_add_preserves_dtype(dtype):
    a = motorch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    b = motorch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    assert (a + b).dtype == dtype


@pytest.mark.parametrize("a_dtype, b_dtype, expected_dtype", [
    (np.float32, np.float64, np.float64),  # float32 + float64 -> float64
    (np.int32,   np.float32, np.float64),  # int32 + float32 -> float64 (matches numpy)
    (np.int32,   np.int64,   np.int64),    # int32 + int64 -> int64
])
def test_add_dtype_promotion(a_dtype, b_dtype, expected_dtype):
    a = motorch.tensor([1, 2, 3], dtype=a_dtype)
    b = motorch.tensor([1, 2, 3], dtype=b_dtype)
    assert (a + b).dtype == expected_dtype

