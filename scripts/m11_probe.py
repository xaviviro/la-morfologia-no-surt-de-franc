"""Linear probing (Edmiston 2020 style): is morphological information linearly
*decodable* from the embeddings, and does morpheme-aware segmentation make it
more decodable?

For two non-trivial-ish features we train a logistic-regression probe with
group-cross-validation (grouped by lemma, so the singular and plural of the same
word never split across train/test) and compare probe accuracy under the native
vs morphemic segmentation of the derived form:

  number : singular (base) vs plural (derived)  — families plural/es_plural/gem_lla/cedilla/ny
  gender : masculine (base) vs feminine (derived) — families gender_a/es_genero_a

This turns "the geometry is cleaner" into "the morphological feature is more
linearly accessible" — a behavioural-ish bridge. Reads the NPZ already produced
by m03 (mention carrier); no GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from scripts.embed_lib import slugify
from scripts.m04_geometry import GEOMETRY_MODELS

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "number": (["plural", "es_plural", "gem_lla", "cedilla", "ny"]),
    "gender": (["gender_a", "es_genero_a"]),
}


def _probe_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> float:  # noqa: N803
    if len(np.unique(y)) < 2:
        return float("nan")
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, te in gkf.split(X, y, groups):
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs))


def run_model(model_id: str) -> list[dict]:
    mdir = ROOT / "out" / slugify(model_id)
    if not (mdir / "metadata.parquet").exists():
        return []
    meta = pd.read_parquet(mdir / "metadata.parquet")
    L = GEOMETRY_MODELS[model_id][-1]
    vecs = np.load(mdir / f"embeddings_layer{L}.npz")["vectors"]
    is_mention = meta.carrier_kind == "mention" if "carrier_kind" in meta.columns else True
    rows = []
    for task, fams in TASKS.items():
        for cond in ("native", "morphemic"):
            # base = label 0 (singular/masc), derived = label 1 (plural/fem)
            sel = meta[(meta.family.isin(fams)) & (meta.condition == cond) & is_mention]
            base = sel[sel.role == "base"]
            der = sel[sel.role == "derived"]
            idx = np.concatenate([base.index.to_numpy(), der.index.to_numpy()])
            X = vecs[idx]
            y = np.concatenate([np.zeros(len(base)), np.ones(len(der))]).astype(int)
            groups = np.concatenate([base["base"].to_numpy(), der["base"].to_numpy()])
            rows.append({"model": model_id, "task": task, "condition": cond,
                         "accuracy": _probe_accuracy(X, y, groups), "n": len(y)})
    return rows


def main() -> None:
    rows = []
    for model_id in GEOMETRY_MODELS:
        rows += run_model(model_id)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "out" / "probe_accuracy.csv", index=False)
    # native vs morphemic summary per task
    print("Linear probe accuracy (grouped CV by lemma), native vs morphemic:")
    for task in TASKS:
        nat = df[(df.task == task) & (df.condition == "native")]["accuracy"].mean()
        mor = df[(df.task == task) & (df.condition == "morphemic")]["accuracy"].mean()
        print(f"  {task:7} native={nat:.3f}  morphemic={mor:.3f}  Δ={mor - nat:+.3f}")
    print("wrote out/probe_accuracy.csv")


if __name__ == "__main__":
    main()
