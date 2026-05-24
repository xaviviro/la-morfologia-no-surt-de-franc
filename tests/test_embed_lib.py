import numpy as np
import pytest

from scripts.embed_lib import (
    gold_boundaries,
    token_cut_positions,
    boundary_recall,
    char_span_to_token_span,
    fertility,
    assemble_ids,
    random_split,
)


def test_random_split_joins_to_word_and_avoids_gold():
    # "ràpidament" gold boundary at 6; random split into 2 pieces must avoid 6
    pieces = random_split("ràpidament", [6], 2)
    assert len(pieces) == 2
    assert "".join(pieces) == "ràpidament"
    assert len(pieces[0]) != 6  # cut is not at the gold boundary


def test_random_split_is_deterministic():
    assert random_split("muntanyes", [6], 2) == random_split("muntanyes", [6], 2)


def test_random_split_single_piece_when_native():
    assert random_split("gat", [], 1) == ["gat"]


def test_gold_boundaries_internal_positions():
    # "ràpida|ment" -> boundary after 6 chars
    assert gold_boundaries("ràpida|ment") == [6]
    # three morphemes -> two internal boundaries
    assert gold_boundaries("cant|av|em") == [4, 6]
    # single morpheme -> none
    assert gold_boundaries("gat") == []


def test_token_cut_positions_from_offsets():
    # offsets for a word tokenized as ["ràpid","ament"] (chars 0-5, 5-10)
    offsets = [(0, 5), (5, 10)]
    assert token_cut_positions(offsets) == {5}
    # single token -> no internal cut
    assert token_cut_positions([(0, 3)]) == set()


def test_boundary_recall():
    assert boundary_recall([6], {6}) == 1.0
    assert boundary_recall([6], {5}) == 0.0
    assert boundary_recall([4, 6], {6}) == 0.5
    assert boundary_recall([], {3}) == 1.0  # vacuously satisfied


def test_char_span_to_token_span_basic():
    # tokens: [BOS(0,0), (0,2), (2,7), (7,8)]; target chars [2,7)
    offsets = [(0, 0), (0, 2), (2, 7), (7, 8)]
    assert char_span_to_token_span(offsets, 2, 7) == (2, 3)


def test_fertility_counts_tokens():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    assert fertility(tok, "cat") >= 1
    # a long Catalan derived word should fragment into >1 piece on gpt2
    assert fertility(tok, "ràpidament") > 1


def test_assemble_ids_native_vs_morphemic_share_prefix_suffix():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    prefix, suffix = "I read the word ", " in a book."
    ids_n, span_n = assemble_ids(tok, prefix, ["ràpidament"], suffix)
    ids_m, span_m = assemble_ids(tok, prefix, ["ràpida", "ment"], suffix)
    # prefix region identical
    assert ids_n[: span_n[0]] == ids_m[: span_m[0]]
    # suffix region identical
    assert ids_n[span_n[1]:] == ids_m[span_m[1]:]
    # word region differs (morphemic forces the split)
    assert ids_n[span_n[0]:span_n[1]] != ids_m[span_m[0]:span_m[1]]
    # spans are non-empty
    assert span_n[1] > span_n[0] and span_m[1] > span_m[0]
