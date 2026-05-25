"""Carrier robustness: does the morphemic gain survive a different carrier frame?

The primary results use a *mention* carrier ("He llegit la paraula {w}..."). A
reviewer concern is that mention frames may be atypical. Here we recompute the
headline delta (oracle − native, direction consistency) under both the mention
carrier and a per-family *use* carrier, per (model, family) at the deepest layer,
and check whether the two agree (sign agreement + correlation). Reads the
metadata `carrier_kind` column produced by the updated m03."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.embed_lib import slugify
from scripts.geom_lib import direction_consistency, sign_test, spearman_r
from scripts.m04_geometry import CA_FAMILIES, GEOMETRY_MODELS, _matrices

ROOT = Path(__file__).resolve().parents[1]


def _delta_dir(meta, vecs, family, carrier_kind):
    """oracle − native direction-consistency delta for a family under a carrier."""
    bn, dn = _matrices(meta, vecs, family, "native", carrier_kind)
    bm, dm = _matrices(meta, vecs, family, "morphemic", carrier_kind)
    if len(bn) < 2 or len(bm) < 2:
        return float("nan")
    return direction_consistency(dm - bm) - direction_consistency(dn - bn)


def collect(metrics_root: Path = ROOT / "out") -> pd.DataFrame:
    rows = []
    for model, layers in GEOMETRY_MODELS.items():
        mdir = metrics_root / slugify(model)
        meta_path = mdir / "metadata.parquet"
        if not meta_path.exists():
            continue
        meta = pd.read_parquet(meta_path)
        if "carrier_kind" not in meta.columns or "use" not in set(meta.carrier_kind):
            continue
        vecs = np.load(mdir / f"embeddings_layer{layers[-1]}.npz")["vectors"]
        for fam in CA_FAMILIES:
            rows.append({
                "model": model, "family": fam,
                "delta_mention": _delta_dir(meta, vecs, fam, "mention"),
                "delta_use": _delta_dir(meta, vecs, fam, "use"),
            })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["delta_mention", "delta_use"])
    agree = int(np.sum(np.sign(d.delta_mention) == np.sign(d.delta_use)))
    rho = spearman_r(d.delta_mention, d.delta_use)
    # is the use-carrier delta consistently positive too? (sign test on its sign)
    pos_use, n_use, p_use = sign_test(d.delta_use.to_numpy())
    return {
        "n_cells": len(d),
        "sign_agreement": agree,
        "sign_agreement_frac": agree / len(d) if len(d) else float("nan"),
        "spearman": rho,
        "use_positive": pos_use,
        "use_n": n_use,
        "use_sign_p": p_use,
        "mean_delta_mention": float(d.delta_mention.mean()),
        "mean_delta_use": float(d.delta_use.mean()),
    }


def main() -> None:
    df = collect()
    if df.empty:
        print("no use-carrier data found (re-run m03 with the two-carrier loop)")
        return
    out = ROOT / "out" / "carrier_robustness.csv"
    df.to_csv(out, index=False)
    s = summarise(df)
    print(f"wrote {out}  ({len(df)} cells)")
    print(f"  mean delta: mention={s['mean_delta_mention']:+.3f}  use={s['mean_delta_use']:+.3f}")
    print(f"  sign agreement: {s['sign_agreement']}/{s['n_cells']} "
          f"({s['sign_agreement_frac']:.0%})  Spearman={s['spearman']:+.3f}")
    print(f"  use-carrier delta positive: {s['use_positive']}/{s['use_n']} "
          f"(sign test p={s['use_sign_p']:.4f})")


if __name__ == "__main__":
    main()
