"""Guards that bundled data files resolve from the project root.

This is a regression test for the example-data loader, which broke when the
code moved into the package (``__file__`` then pointed at the package dir
instead of the repo root where the sample files live).
"""
import os

import tpp_solver
from tpp_solver.io_utils import read_csv_file, read_tsv_file


def test_project_root_contains_bundled_data():
    root = tpp_solver.PROJECT_ROOT
    assert os.path.isfile(os.path.join(root, "sample_data.tsv"))
    assert os.path.isfile(os.path.join(root, "sample_metadata.csv"))


def test_example_data_loads_from_project_root():
    root = tpp_solver.PROJECT_ROOT
    tsv = read_tsv_file(os.path.join(root, "sample_data.tsv"))
    csv = read_csv_file(os.path.join(root, "sample_metadata.csv"))
    assert len(tsv) > 0
    assert "Protein ID" in tsv.columns
    assert {"Temperature", "Treatment", "Samples"}.issubset(set(csv.columns))
