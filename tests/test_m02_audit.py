import math

import pandas as pd
from transformers import AutoTokenizer

from scripts.m02_tokenize_audit import audit_word, audit_tokenizer, fertility_gap_ci


def test_audit_word_reports_fertility_and_recall():
    tok = AutoTokenizer.from_pretrained("gpt2")
    row = audit_word(tok, "ca", "ment", "ràpidament", "ràpida|ment")
    assert row["fertility"] >= 1
    assert 0.0 <= row["boundary_recall"] <= 1.0
    assert row["family"] == "ment"


def test_audit_word_nan_when_offsets_unusable():
    """A tokenizer whose offsets are byte-based (max_end != len(word) in chars)
    should yield boundary_recall=NaN while fertility stays valid."""

    class FakeTok:
        """Simulates a broken tokenizer that returns byte-based / unusable offsets.

        For 'ràpidament' (10 chars, 11 bytes) the stub returns three tokens all
        starting at 0 with max_end=7 — not equal to len('ràpidament')=10 — which
        is the DeepSeek pattern.  fertility() only reads input_ids, so it works.
        """

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            out = {"input_ids": [0, 1, 2]}
            if return_offsets_mapping:
                # All-zero-start offsets; max_end (7) != len(text) (10) → unusable
                out["offset_mapping"] = [(0, 1), (0, 1), (0, 7)]
            return out

    row = audit_word(FakeTok(), "ca", "ment", "ràpidament", "ràpida|ment")
    assert math.isnan(row["boundary_recall"]), "expected NaN for unusable offsets"
    assert row["fertility"] == 3


def test_audit_tokenizer_returns_one_row_per_word():
    tok = AutoTokenizer.from_pretrained("gpt2")
    df = pd.DataFrame([
        {"lang": "ca", "family": "ment", "derived": "ràpidament", "gold_segmentation": "ràpida|ment"},
        {"lang": "en", "family": "ly", "derived": "quickly", "gold_segmentation": "quick|ly"},
    ])
    out = audit_tokenizer("gpt2", tok, df)
    assert len(out) == 2
    assert set(out["model"]) == {"gpt2"}


def test_fertility_gap_ci_positive_when_ca_longer():
    rows = []
    for w in range(30):
        rows.append({"model": "m", "lang": "ca", "fertility": 3.0})
        rows.append({"model": "m", "lang": "en", "fertility": 1.0})
    gap = fertility_gap_ci(pd.DataFrame(rows), n=300, seed=0)
    assert len(gap) == 1
    r = gap.iloc[0]
    assert r["gap"] == 2.0 and r["ci_lo"] > 0.0 and r["ci_lo"] <= r["ci_hi"]
