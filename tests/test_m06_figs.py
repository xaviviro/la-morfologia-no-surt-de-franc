import pandas as pd

from scripts.m06_figs import (
    fig_delta_heatmap,
    fig_fertility_ratio,
    fig_ment_delta_forest,
    fig_tokenization_examples,
)


def _synth_metrics():
    from scripts.m04_geometry import GEOMETRY_MODELS

    rows = []
    for model, layers in GEOMETRY_MODELS.items():
        L = layers[-1]
        for fam in ("ment", "plural", "ly"):
            for cond in ("native", "morphemic", "delta"):
                row = {"model": model, "layer": L, "family": fam, "condition": cond,
                       "direction_consistency": 0.1, "analogy_acc": 0.1,
                       "pc1_var_ratio": 0.0, "n_pairs": 20}
                if cond == "delta":
                    row.update({"direction_consistency_ci_lo": 0.02,
                                "direction_consistency_ci_hi": 0.18,
                                "analogy_acc_ci_lo": -0.05, "analogy_acc_ci_hi": 0.15})
                rows.append(row)
    return pd.DataFrame(rows)


def test_fig_fertility_ratio_writes_png(tmp_path):
    audit = pd.DataFrame({
        "model": ["google/gemma-2-2b", "google/gemma-2-2b", "BSC-LT/salamandra-2b", "BSC-LT/salamandra-2b"],
        "lang": ["ca", "en", "ca", "en"],
        "family": ["ment", "ly", "ment", "ly"],
        "derived": ["ràpidament", "quickly", "ràpidament", "quickly"],
        "fertility": [2.4, 1.2, 1.6, 1.15],
        "boundary_recall": [0.2, 0.1, 0.2, 0.1],
        "n_gold_boundaries": [1, 1, 1, 1],
    })
    p = tmp_path / "ratio.png"
    fig_fertility_ratio(audit, p)
    assert p.exists() and p.stat().st_size > 0


def test_fig_tokenization_examples_writes_png(tmp_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    p = tmp_path / "examples.png"
    fig_tokenization_examples([("ràpidament", "ràpida|ment"), ("gatet", "gat|et")],
                              [("gpt2", tok)], p)
    assert p.exists() and p.stat().st_size > 0


def test_fig_delta_heatmap_writes_png(tmp_path):
    p = tmp_path / "heat.png"
    fig_delta_heatmap(_synth_metrics(), "direction_consistency", p)
    assert p.exists() and p.stat().st_size > 0


def test_fig_ment_delta_forest_writes_png(tmp_path):
    p = tmp_path / "forest.png"
    fig_ment_delta_forest(_synth_metrics(), p)
    assert p.exists() and p.stat().st_size > 0
