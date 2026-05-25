import pandas as pd

from scripts.rule_seg import agreement_with_gold, boundary_set, segment


def test_segment_known_affixes():
    assert segment("ràpidament") == ["ràpida", "ment"]
    assert segment("gatet") == ["gat", "et"]
    assert segment("desfer") == ["des", "fer"]
    assert segment("inútil") == ["in", "útil"]


def test_segment_joins_to_word():
    for w in ["educació", "muntanyes", "treballador", "globalització"]:
        assert "".join(segment(w)) == w


def test_segment_short_word_left_whole():
    assert segment("sol") == ["sol"]  # too short to cut and keep MIN_STEM


def test_boundary_set():
    assert boundary_set(["des", "fer"]) == {3}
    assert boundary_set(["sencer"]) == set()


def test_agreement_with_gold_runs():
    df = pd.DataFrame([
        {"lang": "ca", "family": "ment", "derived": "ràpidament",
         "gold_segmentation": "ràpida|ment"},
        {"lang": "ca", "family": "pre_des", "derived": "desfer",
         "gold_segmentation": "des|fer"},
    ])
    rows = agreement_with_gold(df)
    assert len(rows) == 2
    # both are exact matches with the rule segmenter
    assert all(r["recall"] == 1.0 for r in rows)
