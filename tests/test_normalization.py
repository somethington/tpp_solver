"""Tests for the pure normalization helpers."""
import numpy as np
import pytest

from tpp_solver import normalization as m


def test_median_basic_and_zero_preservation():
    values = np.array([0.0, 1.0, 2.0, 4.0])
    out = m.normalize_by_median(values)
    # positive median is 2.0 -> nonzero values divided by 2, zeros preserved
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0, 2.0])


def test_median_all_zero_returns_unchanged():
    values = np.zeros(5)
    np.testing.assert_array_equal(m.normalize_by_median(values), values)


def test_robust_zscore_centers_on_median():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = m.normalize_by_robust_zscore(values)
    # median is 3 -> the middle element maps to 0 and the result is symmetric
    assert out[2] == pytest.approx(0.0)
    np.testing.assert_allclose(out, -out[::-1], atol=1e-12)


def test_robust_zscore_constant_input_is_safe():
    values = np.full(4, 7.0)  # MAD == 0 -> function must not divide by zero
    np.testing.assert_array_equal(m.normalize_by_robust_zscore(values), values)


def test_winsorization_caps_extremes_and_preserves_zeros():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 100.0])
    out = m.normalize_by_winsorization(values, limits=(0.2, 0.2))
    assert out[0] == 0.0                 # zero preserved
    assert out.max() < 100.0             # extreme value pulled in
    assert len(out) == len(values)


def test_quantile_preserves_length_and_zeros():
    values = np.array([0.0, 5.0, 3.0, 9.0, 1.0])
    out = m.normalize_by_quantile(values)
    assert out[0] == 0.0
    assert len(out) == len(values)


@pytest.mark.parametrize(
    "fn",
    [
        m.normalize_by_median,
        m.normalize_by_robust_zscore,
        m.normalize_by_quantile,
        m.normalize_by_winsorization,
    ],
)
def test_empty_input_returns_empty(fn):
    out = fn(np.array([]))
    assert len(out) == 0
