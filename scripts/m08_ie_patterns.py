"""IE-motivated pattern analyses (see docs/morphology-background.md):

  B/C  Sturtevant regularity gradient: native direction consistency should decay
       regular > stem-alternation > suppletive (verbs, 1sg present).
  A    Prefixation vs suffixation: do prefixes give a cleaner morphemic delta
       than suffixes (less phonological fusion / allomorphy)?
  D    Derivational depth: does -ció consistency drop when the base is already
       derived (nom_cio_d1) vs a simple base (nom_cio)?

Reads out/geometry_metrics.csv (produced by m04). Emits out/ie_patterns_*.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.geom_lib import bootstrap_diff_ci, bootstrap_mean_ci
from scripts.m04_geometry import GEOMETRY_MODELS, VERB_GRADIENT

ROOT = Path(__file__).resolve().parents[1]
DEEP = {m: ls[-1] for m, ls in GEOMETRY_MODELS.items()}
PREFIX_FAMS = ["pre_des", "pre_re", "pre_in"]
SUFFIX_DERIV_FAMS = ["ment", "dim_et", "agent_dor", "nom_cio"]


def _cell_values(metrics, families, condition, metric):
    """One value per (model, family): the metric at the deepest layer."""
    vals = []
    for model, L in DEEP.items():
        for fam in families:
            sub = metrics[(metrics.model == model) & (metrics.layer == L)
                          & (metrics.family == fam) & (metrics.condition == condition)]
            if len(sub):
                v = float(sub[metric].iloc[0])
                if not np.isnan(v):
                    vals.append(v)
    return np.array(vals)


def sturtevant_gradient(metrics) -> pd.DataFrame:
    """Native direction consistency per regularity level, with bootstrap CI."""
    rows = []
    for fam in VERB_GRADIENT:
        vals = _cell_values(metrics, [fam], "native", "direction_consistency")
        mean, lo, hi = bootstrap_mean_ci(vals, n=1000, seed=0)
        rows.append({"level": fam, "native_consistency": mean,
                     "ci_lo": lo, "ci_hi": hi, "n": len(vals)})
    return pd.DataFrame(rows)


def prefix_vs_suffix(metrics) -> dict:
    """Morphemic-native delta (direction): prefixes vs derivational suffixes."""
    pre = _cell_values(metrics, PREFIX_FAMS, "delta", "direction_consistency")
    suf = _cell_values(metrics, SUFFIX_DERIV_FAMS, "delta", "direction_consistency")
    mean, lo, hi, p = bootstrap_diff_ci(pre, suf, n=1000, seed=0)
    return {"prefix_mean_delta": float(np.mean(pre)), "suffix_mean_delta": float(np.mean(suf)),
            "diff": mean, "ci_lo": lo, "ci_hi": hi, "p": p,
            "n_prefix": len(pre), "n_suffix": len(suf)}


def depth_effect(metrics) -> dict:
    """Morphemic direction consistency of -ció on a simple base (d0=nom_cio) vs a
    derived base (d1=nom_cio_d1)."""
    d0 = _cell_values(metrics, ["nom_cio"], "morphemic", "direction_consistency")
    d1 = _cell_values(metrics, ["nom_cio_d1"], "morphemic", "direction_consistency")
    mean, lo, hi, p = bootstrap_diff_ci(d0, d1, n=1000, seed=0)
    return {"depth0_mean": float(np.mean(d0)), "depth1_mean": float(np.mean(d1)),
            "diff_d0_minus_d1": mean, "ci_lo": lo, "ci_hi": hi, "p": p}


def main() -> None:
    metrics = pd.read_csv(ROOT / "out" / "geometry_metrics.csv")

    grad = sturtevant_gradient(metrics)
    grad.to_csv(ROOT / "out" / "ie_sturtevant_gradient.csv", index=False)
    print("Sturtevant gradient (native direction consistency):")
    for r in grad.itertuples():
        print(f"  {r.level:10} {r.native_consistency:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]")

    pvs = prefix_vs_suffix(metrics)
    depth = depth_effect(metrics)
    pd.DataFrame([{"analysis": "prefix_vs_suffix", **pvs},
                  {"analysis": "depth_effect", **depth}]).to_csv(
        ROOT / "out" / "ie_patterns_summary.csv", index=False)
    print(f"Prefix Δ={pvs['prefix_mean_delta']:+.3f} vs suffix Δ={pvs['suffix_mean_delta']:+.3f}"
          f"  (diff {pvs['diff']:+.3f} [{pvs['ci_lo']:+.3f},{pvs['ci_hi']:+.3f}], p={pvs['p']:.3f})")
    print(f"Depth: -ció d0={depth['depth0_mean']:.3f} vs d1={depth['depth1_mean']:.3f}"
          f"  (diff {depth['diff_d0_minus_d1']:+.3f} [{depth['ci_lo']:+.3f},{depth['ci_hi']:+.3f}],"
          f" p={depth['p']:.3f})")


if __name__ == "__main__":
    main()
