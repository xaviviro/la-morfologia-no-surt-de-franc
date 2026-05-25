import numpy as np
import pandas as pd

from scripts.m08_ie_patterns import (
    depth_effect,
    prefix_vs_suffix,
    sturtevant_gradient,
)


def _synth(reg=0.62, alt=0.52, supl=0.42, pre_delta=0.10, suf_delta=0.04,
           cio_d0=0.60, cio_d1=0.50):
    from scripts.m04_geometry import GEOMETRY_MODELS

    rows = []
    for model, layers in GEOMETRY_MODELS.items():
        L = layers[-1]

        def add(fam, cond, metric_val):
            rows.append({"model": model, "layer": L, "family": fam, "condition": cond,
                         "direction_consistency": metric_val, "analogy_acc": 0.5,
                         "pc1_var_ratio": 0.1, "n_pairs": 12})

        add("verb_reg", "native", reg)
        add("verb_alt", "native", alt)
        add("verb_supl", "native", supl)
        for f in ("pre_des", "pre_re", "pre_in"):
            add(f, "delta", pre_delta)
        for f in ("ment", "dim_et", "agent_dor", "nom_cio"):
            add(f, "delta", suf_delta)
        add("nom_cio", "morphemic", cio_d0)
        add("nom_cio_d1", "morphemic", cio_d1)
    return pd.DataFrame(rows)


def test_sturtevant_gradient_decreases():
    grad = sturtevant_gradient(_synth()).set_index("level")["native_consistency"]
    assert grad["verb_reg"] > grad["verb_alt"] > grad["verb_supl"]


def test_prefix_vs_suffix_detects_difference():
    res = prefix_vs_suffix(_synth(pre_delta=0.10, suf_delta=0.04))
    assert res["prefix_mean_delta"] > res["suffix_mean_delta"]
    assert res["diff"] > 0


def test_depth_effect_drop():
    res = depth_effect(_synth(cio_d0=0.60, cio_d1=0.50))
    assert res["depth0_mean"] > res["depth1_mean"]
    assert res["diff_d0_minus_d1"] > 0
