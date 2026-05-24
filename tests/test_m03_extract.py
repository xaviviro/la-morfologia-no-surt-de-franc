import numpy as np

from scripts.m03_extract import extract_pair


def test_extract_pair_native_and_morphemic_shapes():
    from scripts.embed_lib import load_model_and_tokenizer

    tok, model = load_model_and_tokenizer("sshleifer/tiny-gpt2")
    layers = [1, 2]
    carrier = "I read the word {w} in a book."
    out = extract_pair(tok, model, carrier, "ràpid", "ràpidament", "ràpida|ment", layers)
    for cond in ("native", "morphemic"):
        for role in ("base", "derived"):
            for L in layers:
                v = out[(cond, role, L)]
                assert v.ndim == 1 and v.shape[0] == model.config.hidden_size
                assert np.isfinite(v).all()
                assert abs(np.linalg.norm(v) - 1.0) < 1e-4  # L2-normalized
