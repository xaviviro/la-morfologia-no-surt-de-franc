"""Build native (A) and morphemic (B) token streams from the AINA corpus.

Tokenisation is CPU-bound (per-piece id-splicing + rule segmentation), so it runs
across a pool of worker processes — the GPU is unused in this phase by design.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from part2.config import CORPUS, CORPUS_FALLBACK, SEED, TARGET_TOKENS
from scripts.embed_lib import slugify
from scripts.rule_seg import segment as rule_segment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "part2" / "out"
N_WORKERS = min(48, max(1, (os.cpu_count() or 8)))

_TOK = None  # per-worker tokenizer, set by _init_worker


def line_pieces(line: str, morphemic: bool) -> list[str]:
    """Ordered text pieces for one line. If morphemic, each word is rule-segmented;
    the first piece of each word keeps the word-initial space marker."""
    out: list[str] = []
    for w in line.split():
        pieces = rule_segment(w) if morphemic else [w]
        for i, p in enumerate(pieces):
            out.append((" " + p) if i == 0 else p)
    return out


def encode_pieces(tok, pieces: list[str]) -> list[int]:
    """Id-splice a list of text pieces: each piece encoded independently (no special
    tokens), then concatenated. Batched so the fast tokenizer parallelises in Rust —
    identical output to encoding piece-by-piece, ~10-20x faster."""
    if not pieces:
        return []
    enc = tok(pieces, add_special_tokens=False)["input_ids"]
    ids: list[int] = []
    for piece_ids in enc:
        ids.extend(piece_ids)
    return ids


def encode_line(tok, line: str, morphemic: bool) -> list[int]:
    """Token ids for one line (id-spliced). Kept for reference; build() batches."""
    return encode_pieces(tok, line_pieces(line, morphemic))


def _text_field(ex: dict) -> str:
    for k in ("text", "content", "raw_text", "document"):
        if k in ex and isinstance(ex[k], str):
            return ex[k]
    return ""


def _init_worker(model_id: str) -> None:
    """Load a tokenizer once per worker process (after fork, so no oversubscription)."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)


def _encode_chunk(arg: tuple[list[str], bool]) -> list[int]:
    """Worker: encode a chunk of lines to id-spliced tokens (native or morphemic)."""
    lines, morphemic = arg
    pieces: list[str] = []
    for line in lines:
        pieces.extend(line_pieces(line, morphemic))
    return encode_pieces(_TOK, pieces)


def _gather_lines(target_words: int) -> list[str]:
    """Stream the corpus until we have ~target_words words (tokens/word >= 1, so this
    yields at least target_words tokens per condition; trimmed to target at save)."""
    from datasets import load_dataset
    try:
        ds = load_dataset(CORPUS, split="train", streaming=True)
    except Exception as exc:  # noqa: BLE001
        print(f"CATalog failed ({exc}); falling back to {CORPUS_FALLBACK}")
        ds = load_dataset(CORPUS_FALLBACK, split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10000)
    lines: list[str] = []
    nwords = 0
    for ex in ds:
        line = _text_field(ex).strip()
        if not line:
            continue
        lines.append(line)
        nwords += line.count(" ") + 1
        if nwords >= target_words:
            break
    return lines


def build(model_id: str, target_tokens: int = TARGET_TOKENS) -> None:
    slug = slugify(model_id)
    a_path, b_path = OUT / slug / "corpus_A.npy", OUT / slug / "corpus_B.npy"
    if a_path.exists() and b_path.exists() and all(
        len(np.load(p, mmap_mode="r")) >= target_tokens for p in (a_path, b_path)
    ):
        print(f"{slug}: corpus already built ({target_tokens} tokens), skipping", flush=True)
        return
    lines = _gather_lines(target_tokens)
    nchunks = N_WORKERS * 4
    size = max(1, len(lines) // nchunks)
    chunks = [lines[i:i + size] for i in range(0, len(lines), size)]
    print(f"{slug}: {len(lines)} lines -> {len(chunks)} chunks on {N_WORKERS} workers",
          flush=True)
    (OUT / slug).mkdir(parents=True, exist_ok=True)
    with mp.Pool(N_WORKERS, initializer=_init_worker, initargs=(model_id,)) as pool:
        for cond, morphemic in (("A", False), ("B", True)):
            parts = pool.map(_encode_chunk, [(c, morphemic) for c in chunks])
            ids: list[int] = []
            for part in parts:
                ids.extend(part)
            arr = np.array(ids[:target_tokens], dtype=np.int32)
            np.save(OUT / slug / f"corpus_{cond}.npy", arr)
            print(f"{slug} {cond}: {len(arr)} tokens -> corpus_{cond}.npy", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    a = p.parse_args()
    build(a.model, a.target_tokens)


if __name__ == "__main__":
    import os
    main()
    os._exit(0)  # avoid noisy dataset-streaming teardown crash after files are saved
