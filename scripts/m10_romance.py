"""Romance replication: does the Catalan pattern hold in Spanish?

Six Spanish families mirror six Catalan ones (mente↔ment, ción↔ció, dor↔dor,
dim↔dim, plural↔plural, género↔gender). For each pair we compare, at the deepest
layer (mention carrier), the native direction consistency and the morphemic
delta (oracle − native). If Spanish tracks Catalan, the fenomenon is not a
Catalan idiosyncrasy but holds across (two) Indo-European Romance languages —
the basis for the IE-internal framing (not full universality)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.geom_lib import spearman_r

ROOT = Path(__file__).resolve().parents[1]

# Catalan family -> parallel Spanish family
PAIRS = {
    "ment": "es_mente",
    "nom_cio": "es_cion",
    "agent_dor": "es_dor",
    "dim_et": "es_dim",
    "plural": "es_plural",
    "gender_a": "es_genero_a",
}
DEEP = {
    "google/gemma-2-2b": 22, "google/gemma-4-E2B": 22, "Qwen/Qwen2-1.5B": 26,
    "Qwen/Qwen3.5-4B-Base": 32, "BSC-LT/salamandra-2b": 21,
}


def _mean(metrics, family, condition, metric):
    vals = []
    for model, L in DEEP.items():
        sub = metrics[(metrics.model == model) & (metrics.layer == L)
                      & (metrics.family == family) & (metrics.condition == condition)]
        if len(sub):
            v = float(sub[metric].iloc[0])
            if not np.isnan(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def build(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ca, es in PAIRS.items():
        rows.append({
            "pair": f"{ca}↔{es}",
            "ca_native": _mean(metrics, ca, "native", "direction_consistency"),
            "es_native": _mean(metrics, es, "native", "direction_consistency"),
            "ca_delta": _mean(metrics, ca, "delta", "direction_consistency"),
            "es_delta": _mean(metrics, es, "delta", "direction_consistency"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    metrics = pd.read_csv(ROOT / "out" / "geometry_metrics.csv")
    if "es_mente" not in set(metrics.family):
        print("no Spanish families in metrics (re-run m03/m04 with the ES lexicon)")
        return
    df = build(metrics)
    df.to_csv(ROOT / "out" / "romance_comparison.csv", index=False)
    rho_n = spearman_r(df["ca_native"], df["es_native"])
    rho_d = spearman_r(df["ca_delta"], df["es_delta"])
    print("Romance replication (CA vs ES, 6 parallel families):")
    print(df.to_string(index=False))
    print(f"  Spearman native consistency CA~ES = {rho_n:+.3f}")
    print(f"  Spearman morphemic delta     CA~ES = {rho_d:+.3f}")
    print(f"  mean delta: CA={df['ca_delta'].mean():+.3f}  ES={df['es_delta'].mean():+.3f}")


if __name__ == "__main__":
    main()
