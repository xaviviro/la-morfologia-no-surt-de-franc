import numpy as np
import pandas as pd

from scripts.m09_carrier_robustness import summarise


def test_summarise_sign_agreement_and_positivity():
    df = pd.DataFrame({
        "model": ["m"] * 6,
        "family": list("abcdef"),
        "delta_mention": [0.10, 0.08, 0.05, 0.12, 0.03, 0.07],
        "delta_use": [0.09, 0.07, 0.06, 0.10, 0.02, 0.08],
    })
    s = summarise(df)
    assert s["n_cells"] == 6
    assert s["sign_agreement"] == 6          # all same sign (positive)
    assert s["use_positive"] == 6
    assert s["spearman"] > 0.5               # mention and use track each other


def test_summarise_drops_nan():
    df = pd.DataFrame({
        "model": ["m", "m"],
        "family": ["a", "b"],
        "delta_mention": [0.1, np.nan],
        "delta_use": [0.1, 0.2],
    })
    assert summarise(df)["n_cells"] == 1
