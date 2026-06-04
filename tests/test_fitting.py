"""Tests for the curve-fitting model functions and the worker data slicer."""
import numpy as np
import pytest

import tpp_solver_mt as m


def test_sigmoid_midpoint_and_monotonicity():
    a, b, plateau = 1.0, 50.0, 0.0
    # At T == b the logistic term is exactly a/2.
    assert m.sigmoid(b, a, b, plateau) == pytest.approx(a / 2 + plateau)
    temps = np.linspace(30, 70, 50)
    vals = m.sigmoid(temps, a, b, plateau)
    assert np.all(np.diff(vals) > 0)  # strictly increasing for a > 0


def test_paper_sigmoidal_midpoint_is_mean_of_plateaus():
    A1, A2, Tm = 1.0, 0.0, 55.0
    assert m.paper_sigmoidal(Tm, A1, A2, Tm) == pytest.approx((A1 + A2) / 2)


def test_slice_replicate_data_returns_only_one_protein():
    replicate_data = {
        "control": {"P1": {37.0: [1.0]}, "P2": {37.0: [2.0]}},
        "drug": {"P1": {37.0: [3.0]}},
    }
    sliced = m._slice_replicate_data(replicate_data, "P1")
    assert set(sliced.keys()) == {"control", "drug"}
    assert list(sliced["control"].keys()) == ["P1"]
    assert "P2" not in sliced["control"]
    assert sliced["drug"]["P1"] == {37.0: [3.0]}


def test_slice_replicate_data_skips_treatments_without_protein():
    replicate_data = {
        "control": {"P1": {37.0: [1.0]}},
        "drug": {"P2": {37.0: [2.0]}},  # no P1 here
    }
    sliced = m._slice_replicate_data(replicate_data, "P1")
    assert "drug" not in sliced
    assert list(sliced.keys()) == ["control"]


def test_process_protein_replicates_recovers_melting_point():
    """Worker runs without Streamlit and with the explicit-args signature."""
    import matplotlib.pyplot as plt

    temps = [37, 40, 45, 50, 55, 60, 65]
    tm = 52.0
    protein_data = {t: [1.0 / (1.0 + np.exp(t - tm))] for t in temps}  # one replicate
    data_dict = {"control": {"P1": protein_data}}

    args = (
        "P1",
        data_dict,
        ["o", "s", "^", "v"],   # marker_opts
        [50, 75, 100],          # size_opts
        [1.0, 0.8, 0.6],        # alpha_opts
        None,                   # selected_temp
        False,                  # normalize_data
        0.8,                    # r2_threshold
        "Reference Temperature",
        (0.05, 0.05),           # winsor_limits
    )
    protein, fig, summary = m.process_protein_replicates(args)

    assert protein == "P1"
    assert len(summary) == 1
    assert summary[0]["treatment"] == "control"
    assert summary[0]["melting_point"] == pytest.approx(tm, abs=1.0)
    plt.close("all")
