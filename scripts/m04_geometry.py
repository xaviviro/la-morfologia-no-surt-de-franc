"""Compute morphological-subspace geometry per (model, family, condition, layer).

Reads each geometry model's metadata.parquet + embeddings_layer{L}.npz, builds
base/derived matrices per (family, condition), and reports direction
consistency, PC1 variance ratio, and leave-one-out analogy accuracy. Also emits
the native→morphemic delta, with a paired bootstrap 95% CI on the delta of
direction consistency and analogy accuracy (computed at each model's deepest
swept layer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.embed_lib import slugify
from scripts.geom_lib import (
    analogy_accuracy_loo,
    analogy_correct_per_pair,
    bootstrap_delta_ci,
    direction_consistency,
    paired_bootstrap_ci,
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
        for L in layers:
            vecs = np.load(mdir / f"embeddings_layer{L}.npz")["vectors"]
            is_deep = L == layers[-1]
            for family in sorted(meta.family.unique()):
                per_cond, mats = {}, {}
                for condition in ("native", "morphemic"):
                    base, derived = _matrices(meta, vecs, family, condition)
                    mats[condition] = (base, derived)
                    m = family_metrics(base, derived)
                    per_cond[condition] = m
                    rows.append({"model": model_id, "layer": L, "family": family,
                                 "condition": condition, **m})

                base_n, der_n = mats["native"]
                _, der_m = mats["morphemic"]
                # paired bootstrap CI on the delta, only at the headline (deepest) layer
                if is_deep and len(base_n) > 1:
                    # direction: bootstrap the family statistic over resampled pairs
                    dir_lo, dir_hi = bootstrap_delta_ci(_dir_stat, base_n, der_n, der_m,
                                                        n=N_BOOT, seed=0)
                    # analogy: bootstrap the per-pair correctness (robust to duplicate
                    # indices, which would otherwise break nearest-neighbour identity)
                    cn = analogy_correct_per_pair(base_n, der_n)
                    cm = analogy_correct_per_pair(base_n, der_m)
                    ana_lo, ana_hi = paired_bootstrap_ci(cm, cn, n=N_BOOT, seed=0)
                else:
                    dir_lo = dir_hi = ana_lo = ana_hi = float("nan")

                rows.append({
                    "model": model_id, "layer": L, "family": family, "condition": "delta",
                    "direction_consistency": per_cond["morphemic"]["direction_consistency"]
                    - per_cond["native"]["direction_consistency"],
                    "analogy_acc": per_cond["morphemic"]["analogy_acc"]
                    - per_cond["native"]["analogy_acc"],
                    "pc1_var_ratio": per_cond["morphemic"]["pc1_var_ratio"]
                    - per_cond["native"]["pc1_var_ratio"],
                    "n_pairs": per_cond["native"]["n_pairs"],
                    "direction_consistency_ci_lo": dir_lo,
                    "direction_consistency_ci_hi": dir_hi,
                    "analogy_acc_ci_lo": ana_lo,
                    "analogy_acc_ci_hi": ana_hi,
                })
        print(f"computed {model_id}")
    out = pd.DataFrame(rows)
    out_path = ROOT / "out" / "geometry_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
