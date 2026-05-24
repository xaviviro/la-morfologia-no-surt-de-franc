"""Audit how each panel tokenizer segments the morphology lexicon.

No GPU: loads tokenizers only. Per derived word: fertility (subwords/word) and
morpheme-boundary recall (does a token cut fall on the gold boundary?)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from scripts.embed_lib import boundary_recall, fertility, gold_boundaries, token_cut_positions

ROOT = Path(__file__).resolve().parents[1]

PANEL = [
    "google/gemma-2-2b",
    "google/gemma-4-E2B",
    "Qwen/Qwen2-1.5B",
    "Qwen/Qwen3.5-4B-Base",
    "google/gemma-4-26B-A4B",
    "mistralai/Mistral-Small-24B-Base-2501",
    "Qwen/Qwen3.6-35B-A3B",
    "deepseek-ai/deepseek-llm-67b-base",
    "BSC-LT/salamandra-2b",
    "BSC-LT/salamandra-7b",
    "BSC-LT/ALIA-40b",
]


def audit_word(tok, lang: str, family: str, derived: str, gold_segmentation: str) -> dict:
    enc = tok(derived, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    gold = gold_boundaries(gold_segmentation)
    # Some tokenizers (e.g. DeepSeek byte-level BPE) return offsets whose
    # maximum end position does not reach the last character of the word
    # (they are byte-based, not char-based, and have overlapping/wrong spans).
    # Checking max_end == len(derived) is the reliable signal that offsets are
    # in character space and usable for boundary recall.
    max_end = max((e for (_s, e) in offsets), default=0)
    offsets_usable = max_end == len(derived)
    if not offsets_usable:
        recall = float("nan")
    else:
        recall = boundary_recall(gold, token_cut_positions(offsets))
    return {
        "lang": lang,
        "family": family,
        "derived": derived,
        "fertility": fertility(tok, derived),
        "boundary_recall": recall,
        "n_gold_boundaries": len(gold),
    }


def audit_tokenizer(model_id: str, tok, df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"model": model_id, **audit_word(tok, r.lang, r.family, r.derived, r.gold_segmentation)}
        for r in df.itertuples()
    ]
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "morph_pairs.csv")
    parts = []
    for model_id in PANEL:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {model_id}: {exc}")
            continue
        parts.append(audit_tokenizer(model_id, tok, df))
        print(f"audited {model_id}")
    out = pd.concat(parts, ignore_index=True)
    out_path = ROOT / "out" / "tokenize_audit.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
