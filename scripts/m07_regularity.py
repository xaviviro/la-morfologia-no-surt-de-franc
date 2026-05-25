"""Regularity analysis: does native morphological consistency track linguistic
regularity (inflection vs derivation; transparency rank)?

Tests the hypothesis from docs/morphology-background.md §5: a morpheme behaves
like a clean vector direction to the extent it is regular/productive. We take
each Catalan family's NATIVE direction consistency (mean over models, deepest
layer) and relate it to two a-priori linguistic labels (data/family_traits.csv):
inflectional vs derivational, and a 3-level transparency rank."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.geom_lib import bootstrap_diff_ci, spearman_r
from scripts.m04_geometry import CA_FAMILIES, GEOMETRY_MODELS

ROOT = Path(__file__).resolve().parents[1]


def consistency_by_family(metrics: pd.DataFrame, condition: str = "native") -> pd.DataFrame:
    """Mean direction consistency per Catalan family at the deepest layer, for a
    given segmentation condition (native / morphemic / morfessor)."""
    deep = {m: ls[-1] for m, ls in GEOMETRY_MODELS.items()}
    rows = []
    for fam in CA_FAMILIES:
        vals = []
        for model, L in deep.items():
            sub = metrics[(metrics.model == model) & (metrics.layer == L)
                          & (metrics.family == fam) & (metrics.condition == condition)]
            if len(sub):
                vals.append(float(sub["direction_consistency"].iloc[0]))
        if vals:
            rows.append({"family": fam, "consistency": float(np.mean(vals)),
                         "n_models": len(vals)})
    return pd.DataFrame(rows)


# kept for the existing unit test / backwards reference
def native_consistency_by_family(metrics: pd.DataFrame) -> pd.DataFrame:
    out = consistency_by_family(metrics, "native")
    return out.rename(columns={"consistency": "native_consistency"})


def run(metrics: pd.DataFrame, traits: pd.DataFrame, condition: str = "native") -> dict:
    cons = consistency_by_family(metrics, condition)
    df = cons.merge(traits, on="family", how="inner")
    infl = df[df.morph_type == "inflectional"]["consistency"].to_numpy()
    deriv = df[df.morph_type == "derivational"]["consistency"].to_numpy()
    mean_diff, lo, hi, p = bootstrap_diff_ci(infl, deriv, n=1000, seed=0)
    rho = spearman_r(df["transparency_rank"], df["consistency"])
    df = df.copy()
    df["condition"] = condition
    return {
        "condition": condition,
        "table": df.sort_values("consistency", ascending=False),
        "infl_mean": float(np.mean(infl)) if len(infl) else float("nan"),
        "deriv_mean": float(np.mean(deriv)) if len(deriv) else float("nan"),
        "overall_mean": float(df["consistency"].mean()),
        "diff": mean_diff, "diff_ci_lo": lo, "diff_ci_hi": hi, "diff_p": p,
        "spearman_rank_vs_consistency": rho,
    }


def main() -> None:
    metrics = pd.read_csv(ROOT / "out" / "geometry_metrics.csv")
    traits = pd.read_csv(ROOT / "data" / "family_traits.csv")

    tables, summary = [], []
    for cond in ("native", "morphemic", "morfessor"):
        if cond not in set(metrics.condition):
            continue
        res = run(metrics, traits, cond)
        tables.append(res["table"])
        summary.append({k: res[k] for k in (
            "condition", "infl_mean", "deriv_mean", "overall_mean",
            "diff", "diff_ci_lo", "diff_ci_hi", "diff_p",
            "spearman_rank_vs_consistency")})
        print(f"[{cond}] overall={res['overall_mean']:.3f}  "
              f"flexió={res['infl_mean']:.3f}  derivació={res['deriv_mean']:.3f}  "
              f"diff={res['diff']:+.3f} [{res['diff_ci_lo']:+.3f},{res['diff_ci_hi']:+.3f}] "
              f"p={res['diff_p']:.3f}  rho={res['spearman_rank_vs_consistency']:+.3f}")

    out = ROOT / "out" / "regularity_analysis.csv"
    pd.concat(tables, ignore_index=True).to_csv(out, index=False)
    pd.DataFrame(summary).to_csv(ROOT / "out" / "regularity_summary.csv", index=False)
    print(f"wrote {out} and out/regularity_summary.csv")


if __name__ == "__main__":
    main()
