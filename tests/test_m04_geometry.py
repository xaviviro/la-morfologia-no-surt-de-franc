import numpy as np
import pandas as pd

from scripts.m04_geometry import family_metrics


def test_family_metrics_keys_and_ranges():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(10, 16)).astype(np.float32)
    derived = base + np.ones(16, dtype=np.float32)  # clean constant offset
    m = family_metrics(base, derived)
    assert {"direction_consistency", "pc1_var_ratio", "analogy_acc", "n_pairs"} <= set(m)
    assert m["n_pairs"] == 10
    assert m["direction_consistency"] > 0.9
    assert m["analogy_acc"] > 0.9


def test_family_metrics_handles_small_family():
    base = np.eye(4)[:1]
    derived = base + 1.0
    m = family_metrics(base, derived)
    assert m["n_pairs"] == 1
