"""Shared config for Part 2 continued-pretraining."""
from __future__ import annotations

MODELS = {
    "google/gemma-2-2b": 22,
    # "google/gemma-4-E2B": 22,  # excluded: unstable with both adamw_bnb_8bit
    # (kernel error) and adafactor (silent divergence — NaN eval loss); the
    # Gemma 3n/4 PLE+MatFormer architecture in bf16 needs a setup we don't have
    # here. Reported honestly in the writeup rather than burning more compute.
    "Qwen/Qwen2-1.5B": 26,
    "Qwen/Qwen3.5-4B-Base": 32,
    "BSC-LT/salamandra-2b": 21,
}
CORPUS = "projecte-aina/CATalog"           # streamed; fallback below
CORPUS_FALLBACK = "projecte-aina/catalan_textual_corpus"
SEED = 42  # corpus shuffle seed (kept identical across runs so A and B see the same text)
TARGET_TOKENS = 25_000_000
SEQ_LEN = 512
TRAIN_SEEDS = (42, 43, 44)  # per-run training seeds for replication
