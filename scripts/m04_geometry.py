"""Compute morphological-subspace geometry per (model, family, condition, layer).

Conditions: native, morphemic (gold segmentation) and random (placebo — same
number of pieces, arbitrary non-morpheme cut). Reports direction consistency,
PC1 variance ratio and leave-one-out analogy accuracy per condition, plus two
deltas at each model's deepest layer, each with a paired bootstrap 95% CI and a
two-sided bootstrap p-value:

  delta            = morphemic − native   (does morpheme-aware help vs native?)
  delta_vs_random  = morphemic − random   (is the gain morpheme-specific?)

p-values are FDR-corrected (Benjamini–Hochberg) across the deepest-layer cells
per (delta, metric). Per-model and per-family aggregate deltas (over the Catalan
families) get a bootstrap CI in out/geometry_aggregate_ci.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.embed_lib import slugify
from scripts.geom_lib import (
    analogy_accuracy_loo,
    analogy_correct_per_pair,
    benjamini_hochberg,
    bootstrap_delta_ci_p,
    bootstrap_mean_ci,
    direction_consistency,
    paired_bootstrap_ci_p,
    pc1_variance_ratio,
)

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_MODELS = {
    "google/gemma-2-2b": [6, 15, 22],
    "google/gemma-4-E2B": [6, 15, 22],
    "Qwen/Qwen2-1.5B": [7, 17, 26],
    "Qwen/Qwen3.5-4B-Base": [9, 22, 32],
    "BSC-LT/salamandra-2b": [6, 14, 21],
}
CA_FAMILIES = ["ment", "dim_et", "agent_dor", "nom_cio", "plural", "verb_em",
               "gender_a", "gem_lla", "cedilla", "ny"]
DELTAS = {"delta": "native", "delta_vs_random": "random"}
METRICS = ["direction_consistency", "analogy_acc"]
N_BOOT = 1000


def family_metrics(base: np.ndarray, derived: np.ndarray) -> dict:
    offsets = derived - base
    return {
        "n_pairs": len(base),
        "direction_consistency": direction_consistency(offsets) if len(base) > 1 else float("nan"),
        "pc1_var_ratio": pc1_variance_ratio(offsets) if len(base) > 1 else float("nan"),
        "analogy_acc": analogy_accuracy_loo(base, derived),
    }


def _dir_stat(base: np.ndarray, derived: np.ndarray) -> float:
    return direction_consistency(derived - base)


def _matrices(meta: pd.DataFrame, vecs: np.ndarray, family: str, condition: str):
    sel = meta[(meta.family == family) & (meta.condition == condition)]
    base = vecs[sel[sel.role == "base"].index.to_numpy()]
    derived = vecs[sel[sel.role == "derived"].index.to_numpy()]
    return base, derived


def main() -> None:
    rows = []
    for model_id, layers in GEOMETRY_MODELS.items():
        mdir = ROOT / "out" / slugify(model_id)
        if not (mdir / "metadata.parquet").exists():
            print(f"SKIP {model_id}: no extraction artifacts")
            continue
        meta = pd.read_parquet(mdir / "metadata.parquet")
        conditions = [c for c in ("native", "morphemic", "random") if c in set(meta.condition)]
        for L in layers:
            vecs = np.load(mdir / f"embeddings_layer{L}.npz")["vectors"]
            is_deep = L == layers[-1]
            for family in sorted(meta.family.unique()):
                mats, mets = {}, {}
                for cond in conditions:
                    base, derived = _matrices(meta, vecs, family, cond)
                    mats[cond] = (base, derived)
                    mets[cond] = family_metrics(base, derived)
                    rows.append({"model": model_id, "layer": L, "family": family,
                                 "condition": cond, **mets[cond]})
                base = mats["native"][0]
                der = {c: mats[c][1] for c in mats}
                for name, ref in DELTAS.items():
                    if ref not in mets:
                        continue
                    drow = {
                        "model": model_id, "layer": L, "family": family, "condition": name,
                        "direction_consistency": mets["morphemic"]["direction_consistency"]
                        - mets[ref]["direction_consistency"],
                        "analogy_acc": mets["morphemic"]["analogy_acc"] - mets[ref]["analogy_acc"],
                        "pc1_var_ratio": mets["morphemic"]["pc1_var_ratio"]
                        - mets[ref]["pc1_var_ratio"],
                        "n_pairs": mets["native"]["n_pairs"],
                    }
                    if is_deep and len(base) > 1:
                        dl, dh, dp = bootstrap_delta_ci_p(_dir_stat, base, der["morphemic"],
                                                          der[ref], n=N_BOOT, seed=0)
                        cm = analogy_correct_per_pair(base, der["morphemic"])
                        cr = analogy_correct_per_pair(base, der[ref])
                        al, ah, ap = paired_bootstrap_ci_p(cm, cr, n=N_BOOT, seed=0)
                    else:
                        dl = dh = dp = al = ah = ap = float("nan")
                    drow.update({
                        "direction_consistency_ci_lo": dl, "direction_consistency_ci_hi": dh,
                        "direction_consistency_p": dp,
                        "analogy_acc_ci_lo": al, "analogy_acc_ci_hi": ah, "analogy_acc_p": ap,
                    })
                    rows.append(drow)
        print(f"computed {model_id}")

    out = pd.DataFrame(rows)
    deep_layer = {m: ls[-1] for m, ls in GEOMETRY_MODELS.items()}
    is_deep = out.apply(lambda r: r["layer"] == deep_layer.get(r["model"]), axis=1)

    # Benjamini–Hochberg FDR across the deepest-layer cells, per (delta, metric)
    for cond in DELTAS:
        for metric in METRICS:
            qcol = f"{metric}_q"
            if qcol not in out.columns:
                out[qcol] = np.nan
            mask = is_deep & (out["condition"] == cond)
            out.loc[mask, qcol] = benjamini_hochberg(out.loc[mask, f"{metric}_p"].to_numpy())

    out_path = ROOT / "out" / "geometry_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(out)} rows)")

    # Aggregate deltas over the Catalan families, with bootstrap CIs
    agg = []
    for cond in DELTAS:
        for metric in METRICS:
            for model_id in GEOMETRY_MODELS:
                L = deep_layer[model_id]
                vals = out[(out.model == model_id) & (out.layer == L) & (out.condition == cond)
                           & (out.family.isin(CA_FAMILIES))][metric].to_numpy()
                mean, lo, hi = bootstrap_mean_ci(vals, n=N_BOOT, seed=0)
                agg.append({"scope": "model", "key": model_id, "condition": cond,
                            "metric": metric, "mean": mean, "ci_lo": lo, "ci_hi": hi,
                            "n": int(np.sum(~np.isnan(vals)))})
            for fam in CA_FAMILIES:
                vals = []
                for model_id in GEOMETRY_MODELS:
                    L = deep_layer[model_id]
                    v = out[(out.model == model_id) & (out.layer == L) & (out.condition == cond)
                            & (out.family == fam)][metric]
                    if len(v):
                        vals.append(float(v.iloc[0]))
                mean, lo, hi = bootstrap_mean_ci(np.array(vals), n=N_BOOT, seed=0)
                agg.append({"scope": "family", "key": fam, "condition": cond, "metric": metric,
                            "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals)})
    agg_path = ROOT / "out" / "geometry_aggregate_ci.csv"
    pd.DataFrame(agg).to_csv(agg_path, index=False)
    print(f"wrote {agg_path}  ({len(agg)} rows)")


if __name__ == "__main__":
    main()
