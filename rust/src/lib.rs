//! tpp_fit — native sigmoid curve fitting for TPP melting curves.
//!
//! THIS FILE IS AN INTENTIONALLY EMPTY STUB. Implement the crate here.
//! It contains no code on purpose — only the contract you must satisfy so the
//! Rust path is a drop-in replacement for the Python fitting loop.
//!
//! ── Public surface (the ONE function Python imports) ─────────────────────────
//!   fit_sigmoid_batch(temps, values) -> list[Optional[(a, tm, plateau, r2)]]
//!     temps  : list of 1-D float64 arrays (one per curve), contiguous
//!     values : list of 1-D float64 arrays (same lengths, same order)
//!     returns: one result per curve, in order; None for skipped/failed fits
//!
//! ── Per-curve contract (MUST match tpp_solver._fit.fit_sigmoid exactly) ──────
//!   model   : y = a / (1 + exp(-(T - b))) + plateau        (b == Tm)
//!   p0      : [max(values), median(temps), min(values)]
//!   skip    : fewer than 4 points                 -> None
//!   fail    : non-convergence / solver error      -> None
//!   r2      : 1 - ss_res / ss_tot ; if ss_tot == 0 -> 0.0
//!   return  : (a, tm, plateau, r2)  in THAT order
//!
//! ── Suggested structure (see ../README.md for the full plan) ────────────────
//!   - a struct implementing levenberg_marquardt::LeastSquaresProblem for the
//!     sigmoid (residuals = model - data; analytic or numerical Jacobian)
//!   - fn fit_one(t: &[f64], y: &[f64]) -> Option<(f64, f64, f64, f64)>
//!   - #[pyfunction] fit_sigmoid_batch(...):
//!       copy numpy -> owned Vec while holding the GIL, then
//!       py.allow_threads(|| curves.par_iter().map(fit_one).collect())
//!   - #[pymodule] fn tpp_fit(...) registering fit_sigmoid_batch
//!
//! ── Gotchas (see README) ─────────────────────────────────────────────────────
//!   1. Use the nalgebra re-exported by levenberg-marquardt; never a separate dep.
//!   2. Force contiguous float64 on the Python side before crossing the boundary.
//!   3. Copy out of numpy BEFORE allow_threads; no Python objects inside it.
//!
//! TODO: write the implementation. (Validate equivalence against scipy on the
//!       sample data, then benchmark vs the saved Python baseline.)
