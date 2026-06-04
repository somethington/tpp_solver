"""Tests for the pure data-processing helpers in tpp_solver_mt."""
import numpy as np
import pandas as pd
import pytest

import tpp_solver_mt as m


def test_impute_is_deterministic_with_seed_and_fills_zeros():
    df = pd.DataFrame({"s1": [10.0, 0.0, 5.0], "s2": [0.0, 3.0, 0.0]})
    samples = ["s1", "s2"]

    out1 = m.impute_filtered_data(df.copy(), samples, lowest_float=1.0, seed=123)
    out2 = m.impute_filtered_data(df.copy(), samples, lowest_float=1.0, seed=123)

    # Same seed -> identical imputation (reproducible results).
    pd.testing.assert_frame_equal(out1[samples], out2[samples])

    # No zeros remain and imputed values fall in (0, lowest_float).
    assert (out1[samples].to_numpy() > 0).all()
    assert 0 < out1.loc[1, "s1"] < 1.0
    # Original non-zero values are preserved.
    assert out1.loc[0, "s1"] == 10.0


def test_impute_different_seeds_differ():
    df = pd.DataFrame({"s1": [0.0, 0.0, 0.0]})
    a = m.impute_filtered_data(df.copy(), ["s1"], 1.0, seed=1)
    b = m.impute_filtered_data(df.copy(), ["s1"], 1.0, seed=2)
    assert not np.allclose(a["s1"].to_numpy(), b["s1"].to_numpy())


def test_process_summary_tables_aggregates_replicates():
    summary = pd.DataFrame(
        {
            "protein": ["P1", "P1", "P2", "P2"],
            "treatment": ["c", "c", "c", "c"],
            "melting_point": [50.0, 52.0, 60.0, 60.0],
            "R²": [0.90, 0.95, 0.80, 0.82],
            "residuals": ["0,0", "0,0", "0,0", "0,0"],
        }
    )
    replicate_table, averaged_table = m.process_summary_tables(summary)

    assert len(replicate_table) == 4  # replicate table keeps every row
    p1 = averaged_table[averaged_table["protein"] == "P1"].iloc[0]
    assert p1["melting_point"] == pytest.approx(51.0)
    assert p1["num_replicates"] == 2
    assert p1["melting_point_std"] == pytest.approx(np.std([50.0, 52.0], ddof=1))

    p2 = averaged_table[averaged_table["protein"] == "P2"].iloc[0]
    assert p2["melting_point_std"] == pytest.approx(0.0)  # identical replicates


def test_get_replicate_data_builds_nested_structure():
    csv_data = pd.DataFrame(
        {
            "Treatment": ["c", "c", "d", "d"],
            "Temperature": [37, 50, 37, 50],
            "Samples": ["s1", "s2", "s3", "s4"],
        }
    )
    filtered = pd.DataFrame(
        {"Protein ID": ["P1"], "s1": [10.0], "s2": [5.0], "s3": [8.0], "s4": [4.0]}
    )

    rd = m.get_replicate_data(csv_data, filtered)

    assert set(rd.keys()) == {"c", "d"}
    assert rd["c"]["P1"][37.0] == [10.0]
    assert rd["c"]["P1"][50.0] == [5.0]
    assert rd["d"]["P1"][37.0] == [8.0]
    assert rd["d"]["P1"][50.0] == [4.0]
