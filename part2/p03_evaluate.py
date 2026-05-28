"""Evaluate A (native-trained) vs B (morphemic-trained) on Catalan minimal pairs,
across multiple training seeds per (model, condition).

Weights are loaded from the local run dir if present, otherwise pulled from the
private HF repo `{user}/morfo-part2-{slug}-{cond}-s{seed}` (set HF_TOKEN). For each
model we evaluate every seed that has BOTH A and B available and report:
  * the per-(model, seed) accuracies and Δ = acc_B − acc_A;
  * per-model mean Δ across seeds with the within-model sign test;
  * the between-model sign test on the per-model mean Δ.

Each model's A is evaluated under its native segmentation; each model's B under
its morphemic segmentation (each system is evaluated where it was trained).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from part2.config import MODELS, TRAIN_SEEDS
from scripts.embed_lib import load_model_and_tokenizer, slugify
from scripts.geom_lib import sign_test
from scripts.m12_minimal_pairs import evaluate_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "part2" / "out"
MINIMAL_PAIRS = ROOT / "data" / "minimal_pairs.csv"


def _hf_user() -> str:
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN")).whoami()["name"]


def _local_has_weights(d: Path) -> bool:
    return d.is_dir() and (any(d.glob("model*.safetensors")) or any(d.glob("pytorch_model*.bin")))


def _source(slug: str, cond: str, seed: int, user: str) -> str | None:
    """Local dir if it has weights, else the HF repo id, else None."""
    local = OUT / f"{slug}_{cond}_s{seed}"
    if _local_has_weights(local):
        return str(local)
    repo = f"{user}/morfo-part2-{slug}-{cond}-s{seed}"
    from huggingface_hub import HfApi
    try:
        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(repo)
        if any(f.rfilename.endswith(".safetensors") for f in info.siblings):
            return repo
    except Exception:  # noqa: BLE001
        return None
    return None


def _mp_accuracy(src: str, model_id: str, morphemic_eval: bool) -> float:
    df = pd.read_csv(MINIMAL_PAIRS)
    tok, model = load_model_and_tokenizer(src)
    rows = evaluate_model(model_id, df, tok=tok, model=model)
    key = "correct_morphemic" if morphemic_eval else "correct_native"
    return float(np.mean([r[key] for r in rows]))


def main() -> None:
    user = _hf_user()
    per_run = []
    for model_id in MODELS:
        slug = slugify(model_id)
        for seed in TRAIN_SEEDS:
            a_src = _source(slug, "A", seed, user)
            b_src = _source(slug, "B", seed, user)
            if a_src is None or b_src is None:
                print(f"SKIP {model_id} s{seed}: A or B weights unavailable")
                continue
            try:
                acc_a = _mp_accuracy(a_src, model_id, morphemic_eval=False)
                acc_b = _mp_accuracy(b_src, model_id, morphemic_eval=True)
            except Exception as exc:  # noqa: BLE001
                print(f"SKIP {model_id} s{seed}: {type(exc).__name__}: {exc}")
                continue
            delta = acc_b - acc_a
            per_run.append({"model": model_id, "seed": seed,
                            "acc_A_native": acc_a, "acc_B_morphemic": acc_b,
                            "delta_B_minus_A": delta})
            print(f"{model_id} s{seed}: A={acc_a:.3f}  B={acc_b:.3f}  Δ={delta:+.3f}")

    df = pd.DataFrame(per_run)
    df.to_csv(OUT / "part2_results.csv", index=False)
    print(f"\nwrote {OUT / 'part2_results.csv'}  ({len(df)} runs, "
          f"{df['model'].nunique() if len(df) else 0} models)")
    if not len(df):
        return

    agg = (df.groupby("model")
             .agg(mean_delta=("delta_B_minus_A", "mean"),
                  std_delta=("delta_B_minus_A", "std"),
                  n_seeds=("delta_B_minus_A", "count"))
             .reset_index())
    agg.to_csv(OUT / "part2_results_per_model.csv", index=False)
    print("\nPer-model summary (mean Δ across seeds):")
    print(agg.to_string(index=False))

    print("\nWithin-model sign test on per-seed Δ:")
    for model, sub in df.groupby("model"):
        deltas = sub["delta_B_minus_A"].to_numpy()
        pos, n, p = sign_test(deltas)
        print(f"  {model}: {pos}/{n} seeds B>A  p={p:.3f}  Δ={deltas.round(3).tolist()}")

    means = agg["mean_delta"].to_numpy()
    pos, n, p = sign_test(means)
    print(f"\nBetween-model sign test on mean Δ: {pos}/{n} models B>A  p={p:.4f}")


if __name__ == "__main__":
    main()
