"""Tests for the Benjamini-Hochberg FDR correction."""
import numpy as np

from tpp_solver.statistics import benjamini_hochberg


def test_bh_empty_input():
    assert len(benjamini_hochberg([])) == 0


def test_bh_uniform_pvalues_collapse_to_constant():
    # p_i = i/n with rank i -> p*n/i is constant; monotonicity keeps it constant.
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    fdr = benjamini_hochberg(p)
    np.testing.assert_allclose(fdr, 0.05)


def test_bh_known_values_and_order_preserved():
    # Input deliberately unsorted to check order restoration.
    p = [0.5, 0.001]
    fdr = benjamini_hochberg(p)
    # sorted: 0.001 (rank1) -> 0.002 ; 0.5 (rank2) -> 0.5
    np.testing.assert_allclose(fdr, [0.5, 0.002])


def test_bh_is_monotone_non_decreasing_in_p_order():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=50)
    fdr = benjamini_hochberg(p)
    order = np.argsort(p)
    ranked_fdr = fdr[order]
    assert np.all(np.diff(ranked_fdr) >= -1e-12)  # non-decreasing along sorted p


def test_bh_never_exceeds_one_after_clipping_is_callers_job():
    # The raw BH value can exceed 1; the function returns it unclipped.
    fdr = benjamini_hochberg([0.9, 0.95])
    assert fdr.max() >= 0.9
