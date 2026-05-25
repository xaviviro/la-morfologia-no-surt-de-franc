import pandas as pd

from scripts.m12_minimal_pairs import evaluate_model, summarise


def test_evaluate_model_with_tiny():
    from scripts.embed_lib import load_model_and_tokenizer

    tok, model = load_model_and_tokenizer("sshleifer/tiny-gpt2")
    df = pd.DataFrame([
        {"phenomenon": "number", "prefix": "I have two ", "good_focus": "cats",
         "bad_focus": "cat", "suffix": " here.", "good_gold": "cat|s"},
        {"phenomenon": "number", "prefix": "I see three ", "good_focus": "dogs",
         "bad_focus": "dog", "suffix": " there.", "good_gold": "dog|s"},
    ])
    rows = evaluate_model("tiny", df, tok=tok, model=model)
    assert len(rows) == 2
    for r in rows:
        assert r["correct_native"] in (0, 1)
        assert r["correct_morphemic"] in (0, 1)


def test_summarise_shapes():
    df = pd.DataFrame({
        "model": ["m"] * 4,
        "phenomenon": ["number", "number", "gender_art", "gender_art"],
        "good": list("abcd"),
        "correct_native": [1, 0, 1, 1],
        "correct_morphemic": [1, 1, 1, 0],
    })
    s = summarise(df)
    assert set(s.phenomenon) == {"number", "gender_art", "ALL"}
    allrow = s[s.phenomenon == "ALL"].iloc[0]
    assert allrow["n"] == 4
    assert allrow["acc_native"] == 0.75
