import numpy as np
import pandas as pd

from scripts.m07_regularity import native_consistency_by_family, run


def _synth_metrics(infl_high=True):
    """Synthetic metrics: inflection families get higher native consistency."""
    from scripts.m04_geometry import GEOMETRY_MODELS

    fam_cons = {"plural": 0.7, "gender_a": 0.7, "verb_em": 0.68,
                "gem_lla": 0.66, "cedilla": 0.66, "ny": 0.66,
                "dim_et": 0.55, "agent_dor": 0.54, "ment": 0.50, "nom_cio": 0.49}
    rows = []
    for model, layers in GEOMETRY_MODELS.items():
        L = layers[-1]
        for fam, c in fam_cons.items():
            rows.append({"model": model, "layer": L, "family": fam,
                         "condition": "native", "direction_consistency": c,
                         "analogy_acc": 0.5, "pc1_var_ratio": 0.1, "n_pairs": 20})
    return pd.DataFrame(rows)


def _traits():
    return pd.read_csv("data/family_traits.csv")


def test_native_consistency_one_row_per_family():
    out = native_consistency_by_family(_synth_metrics())
    assert len(out) == 10
    assert set(out.columns) >= {"family", "native_consistency", "n_models"}


def test_run_detects_inflection_higher():
    res = run(_synth_metrics(), _traits())
    # inflection synthetic mean (~0.67) > derivation (~0.52)
    assert res["infl_mean"] > res["deriv_mean"]
    assert res["diff"] > 0
    # transparency rank 1=regular; higher rank -> lower consistency => negative rho
    assert res["spearman_rank_vs_consistency"] < 0


def test_traits_cover_all_ca_families():
    from scripts.m04_geometry import CA_FAMILIES

    traits = _traits()
    assert set(CA_FAMILIES) <= set(traits["family"])
    assert set(traits["morph_type"]) <= {"inflectional", "derivational"}
