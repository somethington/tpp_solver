"""Speed benchmarks for the compute hot-paths.

These establish baselines for the planned Rust rewrite. The prime Rust targets
are the sigmoid fitting loop (``test_bench_fit_all_proteins``) and the data
reshape (``test_bench_get_replicate_data``).

Run with:  pytest benchmarks/ --benchmark-only
Excluded from the normal test run (pyproject ``testpaths = ["tests"]``).
"""
import numpy as np
from scipy.optimize import curve_fit

from tpp_solver import preprocessing as P
from tpp_solver.models import sigmoid
from tpp_solver.statistics import benjamini_hochberg


def _fit_all(rd, ref):
    """Pure sigmoid-fit hot loop (no plotting) — mirrors the worker's core math.

    This is the prime Rust-rewrite target: per protein/replicate, normalize to
    the reference temperature, fit the sigmoid, and compute Tm + R2.
    """
    out = []
    for treatment, proteins in rd.items():
        for protein, td in proteins.items():
            temps = np.array(sorted(td.keys()))
            if len(temps) < 4:
                continue
            nrep = min(len(td[t]) for t in temps)
            ref_idx = np.where(temps == ref)[0]
            for r in range(nrep):
                vals = np.array([td[t][r] for t in temps], float)
                if ref_idx.size and vals[ref_idx[0]] != 0:
                    vals = vals / vals[ref_idx[0]]
                try:
                    popt, _ = curve_fit(
                        sigmoid, temps, vals,
                        p0=[vals.max(), float(np.median(temps)), vals.min()], maxfev=10000,
                    )
                except Exception:
                    continue
                fitted = sigmoid(temps, *popt)
                ss_res = float(np.sum((vals - fitted) ** 2))
                ss_tot = float(np.sum((vals - vals.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                out.append((protein, treatment, r, float(popt[1]), r2))
    return out


def test_bench_fit_all_proteins(benchmark, data):
    """Sigmoid curve fitting across all proteins/replicates. PRIME Rust target."""
    result = benchmark(_fit_all, data["rd"], data["ref"])
    assert len(result) > 0


def test_bench_get_replicate_data(benchmark, data):
    """Reshape the intensity matrix into the nested replicate dict. Rust target."""
    result = benchmark(P.get_replicate_data, data["meta"], data["imputed"])
    assert result


def test_bench_impute(benchmark, data):
    """Impute missing values across the intensity matrix."""
    benchmark(
        lambda: P.impute_filtered_data(
            data["filtered"].copy(), data["samples"], data["lowest"], seed=42
        )
    )


def test_bench_filter_and_lowest_float(benchmark, data):
    """Coerce + scan the intensity matrix for the lowest non-zero value."""
    benchmark(P.filter_and_lowest_float, data["tsv"], data["samples"])


def test_bench_benjamini_hochberg(benchmark):
    """FDR correction over a vector of p-values."""
    pvals = np.random.default_rng(0).uniform(0, 1, size=2000)
    benchmark(benjamini_hochberg, pvals)
