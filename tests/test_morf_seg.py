import pandas as pd

from scripts.morf_seg import (
    agreement_with_gold,
    boundary_set,
    segment,
    train_morfessor,
)

WORDS = [
    "ràpid", "ràpidament", "lent", "lentament", "clar", "clarament",
    "gat", "gatet", "gats", "casa", "caseta", "cantar", "cantem", "parlar", "parlem",
    "educar", "educació", "informar", "informació", "muntanya", "muntanyes",
    "col·legi", "col·legis", "plaça", "places",
]


def test_segment_pieces_join_to_word():
    model = train_morfessor(WORDS)
    for w in ["ràpidament", "gatet", "col·legis", "cantem"]:
        assert "".join(segment(model, w)) == w


def test_train_is_deterministic():
    a = train_morfessor(WORDS)
    b = train_morfessor(WORDS)
    for w in WORDS:
        assert segment(a, w) == segment(b, w)


def test_boundary_set_interior_positions():
    assert boundary_set(["gat", "et"]) == {3}
    assert boundary_set(["cant", "av", "em"]) == {4, 6}
    assert boundary_set(["sencer"]) == set()


def test_agreement_with_gold_columns_and_ranges():
    model = train_morfessor(WORDS)
    df = pd.DataFrame([
        {"lang": "ca", "family": "ment", "base": "ràpid", "derived": "ràpidament",
         "gold_segmentation": "ràpida|ment"},
        {"lang": "ca", "family": "dim_et", "base": "gat", "derived": "gatet",
         "gold_segmentation": "gat|et"},
    ])
    out = agreement_with_gold(model, df)
    assert {"precision", "recall", "family", "derived"} <= set(out.columns)
    assert ((out["recall"] >= 0) & (out["recall"] <= 1)).all()
    # gatet -> gat|et is an easy cut Morfessor should get
    assert out[out.derived == "gatet"]["recall"].iloc[0] == 1.0
