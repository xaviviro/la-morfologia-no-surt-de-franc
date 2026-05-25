"""Extract native + morphemic embeddings for each (base, derived) pair.

For the *derived* word we build two conditions sharing identical prefix/suffix
ids: native (one word piece) and morphemic (gold morpheme segmentation). The
*base* word is always single-piece (it is the reference point). Vectors are
mean-pooled over the word region and L2-normalized, at each requested layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.embed_lib import (
    assemble_ids,
    embed_span,
    gold_boundaries,
    load_model_and_tokenizer,
    random_split,
    slugify,
)
from scripts.morf_seg import segment as morf_segment
from scripts.morf_seg import train_morfessor
from scripts.rule_seg import segment as rule_segment

ROOT = Path(__file__).resolve().parents[1]

GEOMETRY_LAYERS = {
    "google/gemma-2-2b": [6, 15, 22],
    "google/gemma-4-E2B": [6, 15, 22],
    "Qwen/Qwen2-1.5B": [7, 17, 26],
    "Qwen/Qwen3.5-4B-Base": [9, 22, 32],
    "BSC-LT/salamandra-2b": [6, 14, 21],
}


def _split_carrier(carrier: str) -> tuple[str, str]:
    prefix, suffix = carrier.split("{w}")
    return prefix, suffix


def extract_pair(tok, model, carrier, base, derived, gold_segmentation, layers,
                 morf_model=None):
    prefix, suffix = _split_carrier(carrier)
    morphs = gold_segmentation.split("|")
    out: dict[tuple[str, str, int], np.ndarray] = {}

    ids_b, span_b = assemble_ids(tok, prefix, [base], suffix)
    base_vecs = embed_span(model, ids_b, span_b, layers)

    ids_dn, span_dn = assemble_ids(tok, prefix, [derived], suffix)
    der_native = embed_span(model, ids_dn, span_dn, layers)

    ids_dm, span_dm = assemble_ids(tok, prefix, morphs, suffix)
    der_morph = embed_span(model, ids_dm, span_dm, layers)

    # placebo control: same number of pieces, but cut at a random non-morpheme boundary
    rand_pieces = random_split(derived, gold_boundaries(gold_segmentation), len(morphs))
    ids_dr, span_dr = assemble_ids(tok, prefix, rand_pieces, suffix)
    der_rand = embed_span(model, ids_dr, span_dr, layers)

    # realistic unsupervised segmenter (Morfessor) — present only if a model is given
    der_morf = None
    if morf_model is not None:
        morf_pieces = morf_segment(morf_model, derived) or [derived]
        ids_df, span_df = assemble_ids(tok, prefix, morf_pieces, suffix)
        der_morf = embed_span(model, ids_df, span_df, layers)

    # realistic rule-based Catalan segmenter (deployable, stronger than Morfessor)
    rule_pieces = rule_segment(derived) or [derived]
    ids_dru, span_dru = assemble_ids(tok, prefix, rule_pieces, suffix)
    der_rule = embed_span(model, ids_dru, span_dru, layers)

    for L in layers:
        out[("native", "base", L)] = base_vecs[L]
        out[("morphemic", "base", L)] = base_vecs[L]  # base shared across conditions
        out[("random", "base", L)] = base_vecs[L]
        out[("rules", "base", L)] = base_vecs[L]
        out[("native", "derived", L)] = der_native[L]
        out[("morphemic", "derived", L)] = der_morph[L]
        out[("random", "derived", L)] = der_rand[L]
        out[("rules", "derived", L)] = der_rule[L]
        if der_morf is not None:
            out[("morfessor", "base", L)] = base_vecs[L]
            out[("morfessor", "derived", L)] = der_morf[L]
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default=None, help="comma list; default = panel layers")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else GEOMETRY_LAYERS[args.model]
    )
    out_dir = Path(args.out) if args.out else ROOT / "out" / slugify(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ROOT / "data" / "morph_pairs.csv")
    tok, model = load_model_and_tokenizer(args.model)

    morf_model = train_morfessor(sorted(set(df["base"]) | set(df["derived"])))

    per_layer: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    meta_rows: list[dict] = []
    # two carrier frames: mention (primary) and use (robustness)
    carriers = [("mention", "carrier"), ("use", "carrier_use")]
    for r in df.itertuples():
        for carrier_kind, carrier_col in carriers:
            carrier = getattr(r, carrier_col)
            pair = extract_pair(tok, model, carrier, r.base, r.derived,
                                r.gold_segmentation, layers, morf_model=morf_model)
            for cond in ("native", "morphemic", "random", "morfessor", "rules"):
                for role in ("base", "derived"):
                    for L in layers:
                        per_layer[L].append(pair[(cond, role, L)])
                    meta_rows.append({
                        "lang": r.lang, "family": r.family, "morph_type": r.morph_type,
                        "base": r.base, "derived": r.derived,
                        "condition": cond, "role": role, "carrier_kind": carrier_kind,
                    })

    meta = pd.DataFrame(meta_rows)
    meta.to_parquet(out_dir / "metadata.parquet", index=False)

    for L in layers:
        arr = np.stack(per_layer[L], axis=0)
        np.savez_compressed(out_dir / f"embeddings_layer{L}.npz", vectors=arr)
        print(f"wrote {out_dir}/embeddings_layer{L}.npz  shape={arr.shape}")

    (out_dir / "run_meta.json").write_text(json.dumps({"model": args.model, "layers": layers}, indent=2))


if __name__ == "__main__":
    main()
