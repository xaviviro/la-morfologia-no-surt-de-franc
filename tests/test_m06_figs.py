import pandas as pd

from scripts.m06_figs import (
    fig_condition_ladder,
    fig_delta_heatmap,
    fig_fertility_ratio,
    fig_layer_robustness,
    fig_ment_delta_forest,
    fig_morfessor_agreement,
    fig_placebo,
    fig_tokenization_examples,
)


def _synth_aggregate():
    fams = ["ment", "plural", "gem_lla", "cedilla", "ny"]
    rows = []
    for cond in ("delta", "delta_vs_random"):
        for f in fams:
            rows.append({"scope": "family", "key": f, "condition": cond,
                         "metric": "direction_consistency", "mean": 0.08,
                         "ci_lo": 0.02, "ci_hi": 0.14, "n": 5})
    return pd.DataFrame(rows)


def test_fig_placebo_writes_png(tmp_path):
    p = tmp_path / "placebo.png"
    fig_placebo(_synth_aggregate(), "direction_consistency", p)
    assert p.exists() and p.stat().st_size > 0


def _synth_metrics():
    from scripts.m04_geometry import GEOMETRY_MODELS

    rows = []
    for model, layers in GEOMETRY_MODELS.items():
        for L in layers:  # all layers, for cross-layer robustness figure
            for fam in ("ment", "plural", "ly"):
                for cond in ("native", "morphemic", "random", "morfessor",
                             "delta", "delta_vs_random", "delta_morfessor"):
                    row = {"model": model, "layer": L, "family": fam, "condition": cond,
                           "direction_consistency": 0.1, "analogy_acc": 0.1,
                           "pc1_var_ratio": 0.0, "n_pairs": 20}
                    if cond.startswith("delta"):
                        row.update({"direction_consistency_ci_lo": 0.02,
                                    "direction_consistency_ci_hi": 0.18,
                                    "direction_consistency_q": 0.01,
                                    "analogy_acc_ci_lo": -0.05, "analogy_acc_ci_hi": 0.15,
                                    "analogy_acc_q": 0.2})
                    rows.append(row)
    return pd.DataFrame(rows)


def test_fig_layer_robustness_writes_png(tmp_path):
    p = tmp_path / "layers.png"
    fig_layer_robustness(_synth_metrics(), "direction_consistency", p)
    assert p.exists() and p.stat().st_size > 0


def test_fig_condition_ladder_writes_png(tmp_path):
    p = tmp_path / "ladder.png"
    fig_condition_ladder(_synth_metrics(), "direction_consistency", p)
    assert p.exists() and p.stat().st_size > 0


def test_fig_morfessor_agreement_writes_png(tmp_path):
    agree = pd.DataFrame({
        "family": ["ment", "plural", "gem_lla", "ny", "cedilla"],
        "derived": ["ràpidament", "gats", "col·legis", "anys", "places"],
        "recall": [0.5, 1.0, 0.8, 0.3, 0.0], "precision": [0.5, 1.0, 0.8, 0.3, 0.0],
        "n_pred_cuts": [1, 1, 1, 1, 0], "n_gold_cuts": [1, 1, 1, 1, 1],
    })
    p = tmp_path / "morf.png"
    fig_morfessor_agreement(agree, p)
    assert p.exists() and p.stat().st_size > 0


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
