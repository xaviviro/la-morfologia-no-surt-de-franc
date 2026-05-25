"""Tokenization + hidden-state extraction helpers.

Pure helpers (offset/boundary math) are unit-tested directly. Model loading
and the forward pass mirror the coca study's 02_extract_embeddings.py:
fast tokenizer, mean-pool over the target span, L2-normalize, bf16,
device_map="auto", no quantization.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import torch


def slugify(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-").lower()


def gold_boundaries(gold_segmentation: str) -> list[int]:
    """Internal character boundaries implied by a piped gold segmentation.

    "ràpida|ment" -> [6]; "cant|av|em" -> [4, 6]; "gat" -> [].
    """
    morphs = gold_segmentation.split("|")
    positions: list[int] = []
    acc = 0
    for m in morphs[:-1]:
        acc += len(m)
        positions.append(acc)
    return positions


def token_cut_positions(offsets: list[tuple[int, int]]) -> set[int]:
    """Internal char positions where a token boundary falls, for a word
    tokenized alone. Excludes 0 and the word length (the outer edges)."""
    cuts: set[int] = set()
    starts = [s for (s, e) in offsets if not (s == 0 and e == 0)]
    ends = [e for (s, e) in offsets if not (s == 0 and e == 0)]
    if not ends:
        return cuts
    word_len = max(ends)
    for pos in set(starts) | set(ends):
        if 0 < pos < word_len:
            cuts.add(pos)
    return cuts


def boundary_recall(gold: list[int], token_cuts: set[int]) -> float:
    """Fraction of gold morpheme boundaries that coincide with a token cut.
    No gold boundaries (single morpheme) is vacuously 1.0."""
    if not gold:
        return 1.0
    hit = sum(1 for b in gold if b in token_cuts)
    return hit / len(gold)


def char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    """Map a character span to a token span via offset_mapping (from coca)."""
    t_start: int | None = None
    t_end: int | None = None
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if e <= char_start:
            continue
        if s >= char_end:
            break
        if t_start is None:
            t_start = i
        t_end = i + 1
    if t_start is None or t_end is None:
        raise RuntimeError(f"could not map [{char_start},{char_end}) to {offsets}")
    return t_start, t_end


def fertility(tokenizer, word: str) -> int:
    """Number of subword tokens for `word` in isolation (no special tokens)."""
    return len(tokenizer(word, add_special_tokens=False)["input_ids"])


def assemble_ids(
    tokenizer, prefix: str, word_pieces: list[str], suffix: str
) -> tuple[list[int], tuple[int, int]]:
    """Build input_ids = [bos?] + prefix + word_pieces + suffix, returning the
    token-index range of the word region.

    The first word piece is tokenized with a leading space so both
    SentencePiece (▁) and byte-level BPE (Ġ) get the word-initial marker;
    later morphemes are tokenized bare (word-internal continuation). prefix and
    suffix tokenizations are identical across native/morphemic conditions.
    """
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    piece_ids: list[int] = []
    for i, piece in enumerate(word_pieces):
        text = (" " + piece) if i == 0 else piece
        piece_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
    word_start = len(bos) + len(prefix_ids)
    word_end = word_start + len(piece_ids)
    input_ids = bos + prefix_ids + piece_ids + suffix_ids
    return input_ids, (word_start, word_end)


def random_split(word: str, gold_positions: list[int], n_pieces: int,
                 seed_salt: int = 0) -> list[str]:
    """Split `word` into `n_pieces` at random interior positions that AVOID the
    gold morpheme boundaries — the placebo control for the morphemic condition
    (same number of pieces, arbitrary cut). Deterministic per word (seeded by a
    stable hash of the word) so the control is reproducible."""
    n_cuts = n_pieces - 1
    length = len(word)
    if n_cuts < 1 or length < 2:
        return [word]
    gold = set(gold_positions)
    interior = [p for p in range(1, length) if p not in gold]
    if len(interior) < n_cuts:  # too short to avoid gold — allow any interior cut
        interior = list(range(1, length))
    seed = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16) + seed_salt
    rng = np.random.default_rng(seed)
    cuts = sorted(int(c) for c in rng.choice(interior, size=n_cuts, replace=False))
    pieces, prev = [], 0
    for c in cuts:
        pieces.append(word[prev:c])
        prev = c
    pieces.append(word[prev:])
    return pieces


def load_model_and_tokenizer(model_id: str, dtype=torch.bfloat16):
    """Load a fast tokenizer + causal LM for hidden-state extraction."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if not tok.is_fast:
        raise RuntimeError(f"{model_id}: need a fast tokenizer for offsets")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    return tok, model


@torch.inference_mode()
def embed_span(model, input_ids: list[int], span: tuple[int, int], layers: list[int]):
    """Forward `input_ids`, mean-pool hidden states over `span`, L2-normalize.
    Returns {layer: np.ndarray[hidden]}."""
    ids = torch.tensor([input_ids], device=model.device)
    out = model(ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    t0, t1 = span
    result: dict[int, np.ndarray] = {}
    for L in layers:
        vec = hs[L][0, t0:t1, :].float().mean(dim=0)
        vec = vec / vec.norm().clamp(min=1e-12)
        result[L] = vec.detach().cpu().numpy().astype(np.float32)
    return result


@torch.inference_mode()
def sequence_logprob(model, input_ids: list[int]) -> float:
    """Total log-probability the model assigns to `input_ids` under teacher
    forcing: sum of log p(token_i | token_<i) for i >= 1. Used for minimal-pair
    acceptability (BLiMP-style): the model should prefer the grammatical form."""
    ids = torch.tensor([input_ids], device=model.device)
    out = model(ids, use_cache=False)
    logp = torch.log_softmax(out.logits[0].float(), dim=-1)  # [seq, vocab]
    total = 0.0
    for i in range(1, len(input_ids)):
        total += float(logp[i - 1, input_ids[i]])
    return total
