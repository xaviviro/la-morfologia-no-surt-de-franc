"""Morfessor — a realistic (unsupervised) morpheme segmenter, as an alternative
to the gold/oracle segmentation.

The oracle condition shows the upper bound (morphologically perfect cuts).
Morfessor is the lower-effort, deployable counterpart: an unsupervised model
trained only on the *strings* of the lexicon (it never sees the gold
boundaries), so it tells us what a practical morpheme-aware tokenizer would
recover, not just the ceiling.

`corpusweight=1.0` is chosen because lower weights over-segment to characters
and higher weights leave words whole (see docs/methodology.md §4)."""

from __future__ import annotations

import random
from pathlib import Path

import morfessor
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORPUS_WEIGHT = 1.0
SEED = 0


def train_morfessor(words: list[str], corpusweight: float = CORPUS_WEIGHT, seed: int = SEED):
    """Train a Morfessor Baseline model on a list of word *types* (no counts,
    no gold boundaries). Deterministic given the (sorted) input and seed."""
    random.seed(seed)
    model = morfessor.BaselineModel(corpusweight=corpusweight)
    model.load_data([(1, w) for w in sorted(set(words))])
    model.train_batch()
    return model


def segment(model, word: str) -> list[str]:
    """Morpheme pieces for `word` under the trained model (Viterbi)."""
    return list(model.viterbi_segment(word)[0])


def boundary_set(pieces: list[str]) -> set[int]:
    """Interior character cut positions implied by a piece list."""
    cuts, acc = set(), 0
    for p in pieces[:-1]:
        acc += len(p)
        cuts.add(acc)
    return cuts


def agreement_with_gold(model, df: pd.DataFrame) -> pd.DataFrame:
    """Per derived word, precision/recall of Morfessor's cuts against the gold
    morpheme boundary (Catalan rows only)."""
    from scripts.embed_lib import gold_boundaries

    rows = []
    for r in df[df.lang == "ca"].itertuples():
        gold = set(gold_boundaries(r.gold_segmentation))
        pred = boundary_set(segment(model, r.derived))
        tp = len(gold & pred)
        prec = tp / len(pred) if pred else (1.0 if not gold else 0.0)
        rec = tp / len(gold) if gold else 1.0
        rows.append({"family": r.family, "derived": r.derived,
                     "n_pred_cuts": len(pred), "n_gold_cuts": len(gold),
                     "precision": prec, "recall": rec})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "morph_pairs.csv")
    words = sorted(set(df["base"]) | set(df["derived"]))
    model = train_morfessor(words)
    agree = agreement_with_gold(model, df)
    out = ROOT / "out" / "morfessor_agreement.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    agree.to_csv(out, index=False)
    print(f"trained Morfessor on {len(words)} types; wrote {out}")
    print(f"  mean boundary recall vs gold = {agree['recall'].mean():.3f}, "
          f"precision = {agree['precision'].mean():.3f}")


if __name__ == "__main__":
    main()
