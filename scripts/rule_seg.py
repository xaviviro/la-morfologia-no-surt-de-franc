"""A rule-based Catalan morpheme segmenter — the *deployable* counterpart to the
gold oracle and to unsupervised Morfessor.

Unlike a lemmatiser/POS-tagger (FreeLing, Stanza), which returns lemma+features
rather than surface morpheme boundaries, this returns a *surface* segmentation
(like the gold one) by matching a fixed list of productive Catalan affixes. It
encodes *general* linguistic knowledge of Catalan affixes, not the study's gold
boundaries, so it is an honest "rules you could deploy" condition.

One cut per word (our pairs are base+one affix). Prefixes are tried first, then
the longest matching suffix; if nothing matches, the word is left whole."""

from __future__ import annotations

# Productive Catalan prefixes (surface forms). Order: longest first.
PREFIXES = ["des", "re", "im", "in", "pre", "anti", "sobre", "sub"]

# Productive Catalan suffixes (surface forms), longest first so e.g. -ament/-ació
# win over -ment/-ció when applicable. These are general Catalan affixes, not
# derived from the study lexicon.
SUFFIXES = [
    "ació", "ció", "ment", "dora", "dors", "dor", "etes", "eta", "ets", "et",
    "aire", "esa", "isme", "ista",
    "os", "es", "ns", "em", "en", "à",  # inflectional-ish
    "s", "a",
]

MIN_STEM = 3  # don't cut if it leaves a stem shorter than this


def segment(word: str) -> list[str]:
    """Surface morpheme segmentation of `word` by rule. Returns 1 or 2 pieces."""
    w = word
    # prefix first
    for p in PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= MIN_STEM:
            return [p, w[len(p):]]
    # then longest matching suffix
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= MIN_STEM:
            return [w[: len(w) - len(s)], s]
    return [w]


def boundary_set(pieces: list[str]) -> set[int]:
    cuts, acc = set(), 0
    for p in pieces[:-1]:
        acc += len(p)
        cuts.add(acc)
    return cuts


def agreement_with_gold(df) -> list[dict]:
    """Per Catalan derived word: precision/recall of rule cuts vs the gold
    morpheme boundary."""
    from scripts.embed_lib import gold_boundaries

    rows = []
    for r in df[df.lang == "ca"].itertuples():
        gold = set(gold_boundaries(r.gold_segmentation))
        pred = boundary_set(segment(r.derived))
        tp = len(gold & pred)
        prec = tp / len(pred) if pred else (1.0 if not gold else 0.0)
        rec = tp / len(gold) if gold else 1.0
        rows.append({"family": r.family, "derived": r.derived,
                     "precision": prec, "recall": rec,
                     "n_pred_cuts": len(pred), "n_gold_cuts": len(gold)})
    return rows


def main() -> None:
    from pathlib import Path

    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "morph_pairs.csv")
    rows = agreement_with_gold(df)
    out = pd.DataFrame(rows)
    out.to_csv(root / "out" / "rule_seg_agreement.csv", index=False)
    print(f"rule segmenter vs gold: recall={out['recall'].mean():.3f}, "
          f"precision={out['precision'].mean():.3f}")
    for w in ["ràpidament", "educació", "gatet", "muntanyes", "desfer", "inútil"]:
        print(f"  {w} -> {'|'.join(segment(w))}")


if __name__ == "__main__":
    main()
