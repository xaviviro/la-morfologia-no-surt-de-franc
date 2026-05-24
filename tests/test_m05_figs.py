import pandas as pd

from scripts.m05_figs import fig_fertility, fig_boundary_heatmap


def test_fig_fertility_writes_png(tmp_path):
    audit = pd.DataFrame({
        "model": ["gpt2", "gpt2"], "lang": ["ca", "en"],
        "family": ["ment", "ly"], "derived": ["ràpidament", "quickly"],
        "fertility": [3, 2], "boundary_recall": [1.0, 1.0], "n_gold_boundaries": [1, 1],
    })
    p = tmp_path / "fertility.png"
    fig_fertility(audit, p)
    assert p.exists() and p.stat().st_size > 0


def test_fig_boundary_heatmap_writes_png(tmp_path):
    audit = pd.DataFrame({
        "model": ["gpt2", "gpt2"], "lang": ["ca", "ca"],
        "family": ["ment", "plural"], "derived": ["ràpidament", "gats"],
        "fertility": [3, 1], "boundary_recall": [1.0, 1.0], "n_gold_boundaries": [1, 1],
    })
    p = tmp_path / "heat.png"
    fig_boundary_heatmap(audit, p)
    assert p.exists() and p.stat().st_size > 0
