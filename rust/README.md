# tpp_fit — native fit acceleration (scaffold)

Optional Rust extension that replaces the per-curve sigmoid fitting hot-path
(the prime target identified by `benchmarks/`). **This is a scaffold only** — the
Rust implementation lives in `src/lib.rs`, which is intentionally an empty stub.

## Layout

```
rust/
├── Cargo.toml        # crate + dependency declarations (config)
├── pyproject.toml    # maturin build backend
├── .gitignore        # /target, Cargo.lock
└── src/
    └── lib.rs        # <- YOU implement this (currently a commented contract)
```

## The contract

Expose a single Python function that is a drop-in replacement for the Python
fitting loop. It must produce the **same numbers** as the reference below.

| | |
|---|---|
| Function | `fit_sigmoid_batch(temps, values)` |
| `temps` | list of 1-D `float64` arrays (one per curve) |
| `values` | list of 1-D `float64` arrays (same lengths/order) |
| Returns | list of `Optional[(a, tm, plateau, r2)]`, one per curve, in order |
| Model | `y = a / (1 + exp(-(T - b))) + plateau`  (`b` = Tm) |
| Initial guess `p0` | `[max(values), median(temps), min(values)]` |
| Skip | `< 4` points → `None` |
| Failure | non-convergence / solver error → `None` |
| R² | `1 - ss_res/ss_tot`; `ss_tot == 0` → `0.0` |

### Reference implementation (the spec to match — Python)

The Rust output must agree with this within tolerance (Tm `< 1e-2`, R² `< 1e-3`):

```python
from scipy.optimize import curve_fit
from tpp_solver.models import sigmoid
import numpy as np

def fit_sigmoid(temperatures, values):
    if len(temperatures) < 4:
        return None
    try:
        popt, _ = curve_fit(
            sigmoid, temperatures, values,
            p0=[values.max(), np.median(temperatures), values.min()], maxfev=10000,
        )
    except Exception:
        return None
    fitted = sigmoid(temperatures, *popt)
    ss_res = float(np.sum((values - fitted) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(popt[0]), float(popt[1]), float(popt[2]), r2
```

## Build

```bash
pip install maturin
cd rust
maturin develop --release        # builds + installs `tpp_fit` into the active venv
#   or, from repo root:  make rust
```

## Integrating into the app

Wire it behind an optional import so the app keeps working without the extension:

```python
# tpp_solver/_fit.py
try:
    from tpp_fit import fit_sigmoid_batch as _fit_batch   # native
    FIT_BACKEND = "rust"
except ImportError:
    FIT_BACKEND = "python"
    def _fit_batch(temps_list, values_list):
        return [fit_sigmoid(t, v) for t, v in zip(temps_list, values_list)]
```

The caller collects every replicate's `(temps, values)`, calls `_fit_batch` once,
then builds the summary/plots — which also lets the worker drop `multiprocessing`
(parallelism moves into the Rust call).

## Verify

- **Correctness:** add `tests/test_fit_equivalence.py` that runs both backends on
  the sample data and asserts agreement within the tolerances above. Use
  `pytest.importorskip("tpp_fit")` so it skips when the extension isn't built.
- **Speed:** point `benchmarks/test_bench_fit_all_proteins` at `fit_sigmoid_batch`
  and `--benchmark-compare` against the saved Python baseline.

## Gotchas

1. **nalgebra version** — use the one re-exported by `levenberg-marquardt`
   (`levenberg_marquardt::nalgebra`); a second, mismatched `nalgebra` dependency
   causes opaque trait-bound errors.
2. **Contiguity/dtype** — slicing a numpy array as `&[f64]` requires it to be
   contiguous `float64`; force `np.ascontiguousarray(x, dtype=np.float64)` before
   the call.
3. **GIL** — copy numpy data into owned `Vec`s *before* releasing the GIL
   (`py.allow_threads`); you can't touch Python objects inside the parallel region.

## Notes

- Expect a solid multiple, not orders of magnitude: each curve is only 8 points,
  so the win is removing per-fit Python overhead + lighter parallelism (rayon
  threads vs. Python processes + figure pickling), not faster linear algebra.
- This crate is not part of the app's CI build; it's an optional accelerator.
