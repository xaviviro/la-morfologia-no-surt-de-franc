"""Training diagnostics across seeds: LR, train/eval loss, grad-norm curves;
A vs B per model with one transparent line per seed."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from part2.config import MODELS  # noqa: E402
from scripts.embed_lib import slugify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "part2" / "out"
FIGS = OUT / "figs"
RUN_RE = re.compile(r"(.+)_([AB])_s(\d+)$")


def _seed_dirs(slug: str, cond: str) -> list[tuple[int, Path]]:
    """[(seed, dir)] sorted by seed for one (model, condition)."""
    out = []
    for d in OUT.glob(f"{slug}_{cond}_s*"):
        m = RUN_RE.match(d.name)
        if m and (d / "train_log.csv").exists():
            out.append((int(m.group(3)), d))
    return sorted(out)


def curves_for_model(model_id: str) -> None:
    slug = slugify(model_id)
    runs = {"A": _seed_dirs(slug, "A"), "B": _seed_dirs(slug, "B")}
    if not runs["A"] and not runs["B"]:
        return
    FIGS.mkdir(parents=True, exist_ok=True)
    panels = [("loss", "train loss"), ("learning_rate", "LR"),
              ("grad_norm", "grad norm"), ("eval_loss", "eval loss")]
    palette = {"A": "#bdbdbd", "B": "#1b9e77"}
    label = {"A": "A nadiu", "B": "B morfèmic"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, title) in zip(axes.ravel(), panels):
        for cond in ("A", "B"):
            for i, (seed, d) in enumerate(runs[cond]):
                df = pd.read_csv(d / "train_log.csv")
                if col not in df.columns:
                    continue
                sub = df.dropna(subset=[col])
                if "step" not in sub.columns or not len(sub):
                    continue
                # one entry per condition in the legend, even with multiple seeds
                lbl = label[cond] if i == 0 else None
                ax.plot(sub["step"], sub[col], color=palette[cond], alpha=0.6, lw=1.0, label=lbl)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.legend()
    n_seeds = len({s for cond in ("A", "B") for s, _ in runs[cond]})
    fig.suptitle(f"Entrenament — {slug} (A nadiu vs B morfèmic · {n_seeds} llavors)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGS / f"train_{slug}.png", dpi=150)
    plt.close()


def main() -> None:
    summary = []
    for model_id in MODELS:
        curves_for_model(model_id)
        slug = slugify(model_id)
        for cond in ("A", "B"):
            for _seed, d in _seed_dirs(slug, cond):
                mp = d / "train_meta.json"
                if mp.exists():
                    summary.append(json.loads(mp.read_text()))
    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(OUT / "train_summary.csv", index=False)
        piv = (df.groupby(["model", "condition"])["final_perplexity"]
                 .agg(["mean", "std", "count"]).round(3))
        print("Final perplexity per (model, condition) — mean ± std across seeds:")
        print(piv.to_string())
        print(f"\nwrote {OUT / 'train_summary.csv'} + curves in {FIGS}")


if __name__ == "__main__":
    main()
