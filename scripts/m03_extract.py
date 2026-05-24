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

from scripts.embed_lib import assemble_ids, embed_span, load_model_and_tokenizer, slugify

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


def extract_pair(tok, model, carrier, base, derived, gold_segmentation, layers):
    prefix, suffix = _split_carrier(carrier)
    morphs = gold_segmentation.split("|")
    out: dict[tuple[str, str, int], np.ndarray] = {}

    ids_b, span_b = assemble_ids(tok, prefix, [base], suffix)
    base_vecs = embed_span(model, ids_b, span_b, layers)

    ids_dn, span_dn = assemble_ids(tok, prefix, [derived], suffix)
    der_native = embed_span(model, ids_dn, span_dn, layers)

    ids_dm, span_dm = assemble_ids(tok, prefix, morphs, suffix)
    der_morph = embed_span(model, ids_dm, span_dm, layers)

    for L in layers:
        out[("native", "base", L)] = base_vecs[L]
        out[("morphemic", "base", L)] = base_vecs[L]  # base shared across conditions
        out[("native", "derived", L)] = der_native[L]
        out[("morphemic", "derived", L)] = der_morph[L]
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

    per_layer: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    meta_rows: list[dict] = []
    for r in df.itertuples():
        pair = extract_pair(tok, model, r.carrier, r.base, r.derived,
                            r.gold_segmentation, layers)
        for cond in ("native", "morphemic"):
            for role in ("base", "derived"):
                for L in layers:
                    per_layer[L].append(pair[(cond, role, L)])
                meta_rows.append({
                    "lang": r.lang, "family": r.family, "morph_type": r.morph_type,
                    "base": r.base, "derived": r.derived, "condition": cond, "role": role,
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
