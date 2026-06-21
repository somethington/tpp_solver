"""Shared fixtures for the benchmark suite (loads + preps the sample data once)."""
import os

import numpy as np
import pandas as pd
import pytest

import tpp_solver
from tpp_solver import preprocessing as P

ROOT = tpp_solver.PROJECT_ROOT


@pytest.fixture(scope="session")
def data():
    """Realistic inputs derived from the bundled sample dataset (~1000 proteins)."""
    tsv = pd.read_csv(os.path.join(ROOT, "sample_data.tsv"), sep="\t")
    meta = pd.read_csv(os.path.join(ROOT, "sample_metadata.csv"))[
        ["Temperature", "Treatment", "Samples"]
    ].dropna(subset=["Samples", "Treatment", "Temperature"])
    samples = meta["Samples"].tolist()
    filtered, lowest = P.filter_and_lowest_float(tsv, samples)
    imputed = P.impute_filtered_data(filtered.copy(), samples, lowest, seed=42)
    rd = P.get_replicate_data(meta, imputed)
    ref = float(np.min(pd.to_numeric(meta["Temperature"])))
    return {
        "tsv": tsv,
        "meta": meta,
        "samples": samples,
        "filtered": filtered,
        "lowest": lowest,
        "imputed": imputed,
        "rd": rd,
        "ref": ref,
        "n_proteins": len(tsv),
    }
