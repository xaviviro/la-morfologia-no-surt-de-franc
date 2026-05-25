"""Minimal-pair morphological acceptability (BLiMP-style behavioural probe).

For each pair, the model should assign higher total log-probability to the
grammatical sentence than to the ungrammatical one. We measure accuracy
(% of pairs the model gets right) under two segmentations of the *correct* focus
word: native vs morphemic. If morphemic segmentation makes the model prefer the
correct form more often, the morpheme-aware split helps *behaviour*, not just
geometry.

The grammatical and ungrammatical sentences differ in one focus word
(`good_focus` vs `bad_focus`); the bad form is monomorphemic so it is the fixed
reference. Caveat: morphemic `good` has one extra token, so the length bias is
acknowledged (it applies to both conditions; we compare *accuracies*, not raw
log-probs). Needs GPU."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.embed_lib import assemble_ids, load_model_and_tokenizer, sequence_logprob
from scripts.geom_lib import sign_test
from scripts.m04_geometry import GEOMETRY_MODELS

ROOT = Path(__file__).resolve().parents[1]


def _logp(tok, model, prefix: str, pieces: list[str], suffix: str) -> float:
    ids, _ = assemble_ids(tok, prefix.rstrip(), pieces, suffix)
    return sequence_logprob(model, ids)


def evaluate_model(model_id: str, df: pd.DataFrame, tok=None, model=None) -> list[dict]:
    if model is None:
        tok, model = load_model_and_tokenizer(model_id)
    rows = []
    for r in df.itertuples():
        lp_bad = _logp(tok, model, r.prefix, [r.bad_focus], r.suffix)
        lp_good_nat = _logp(tok, model, r.prefix, [r.good_focus], r.suffix)
        lp_good_mor = _logp(tok, model, r.prefix, str(r.good_gold).split("|"), r.suffix)
        rows.append({
            "model": model_id, "phenomenon": r.phenomenon, "good": r.good_focus,
            "correct_native": int(lp_good_nat > lp_bad),
            "correct_morphemic": int(lp_good_mor > lp_bad),
        })
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for keys, g in df.groupby(["phenomenon"]):
        ph = keys if isinstance(keys, str) else keys[0]
        out.append({"phenomenon": ph, "n": len(g),
                    "acc_native": g.correct_native.mean(),
                    "acc_morphemic": g.correct_morphemic.mean()})
    out.append({"phenomenon": "ALL", "n": len(df),
                "acc_native": df.correct_native.mean(),
                "acc_morphemic": df.correct_morphemic.mean()})
    return pd.DataFrame(out)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "minimal_pairs.csv")
    allrows = []
    for model_id in GEOMETRY_MODELS:
        try:
            allrows += evaluate_model(model_id, df)
            print(f"evaluated {model_id}")
        except Exception as exc:  # noqa: BLE001 — keep going if one model OOMs
            print(f"SKIP {model_id}: {type(exc).__name__}: {exc}")
    res = pd.DataFrame(allrows)
    res.to_csv(ROOT / "out" / "minimal_pairs_raw.csv", index=False)
    summ = summarise(res)
    summ.to_csv(ROOT / "out" / "minimal_pairs_summary.csv", index=False)
    print(summ.to_string(index=False))
    # per-model native vs morphemic accuracy, and a sign test on the per-pair delta
    per_model = res.groupby("model")[["correct_native", "correct_morphemic"]].mean()
    print("\nper model (accuracy native -> morphemic):")
    for m, row in per_model.iterrows():
        print(f"  {m.split('/')[-1]:18} {row.correct_native:.3f} -> {row.correct_morphemic:.3f}")
    delta = res.correct_morphemic - res.correct_native
    pos, n, p = sign_test(delta.to_numpy())
    print(f"\nmorphemic − native per (model,pair): {pos}/{n} positive, sign-test p={p:.4f}")


if __name__ == "__main__":
    main()
