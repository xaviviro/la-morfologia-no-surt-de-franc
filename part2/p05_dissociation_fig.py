"""Part 2 headline figure: the perplexity↔behaviour dissociation.

Left panel  — Δ perplexity (B − A): how much lower B's held-out loss is (B better, left).
Right panel — Δ minimal-pair accuracy (B − A): does that turn into behaviour? (B better, right).
The story: every model gains on perplexity but none gains on behaviour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "part2" / "out"
SHORT = {"google/gemma-2-2b": "gemma-2-2b", "Qwen/Qwen2-1.5B": "Qwen2-1.5B",
         "Qwen/Qwen3.5-4B-Base": "Qwen3.5-4B", "BSC-LT/salamandra-2b": "salamandra-2b"}
ORDER = ["google/gemma-2-2b", "Qwen/Qwen2-1.5B", "Qwen/Qwen3.5-4B-Base", "BSC-LT/salamandra-2b"]


def main() -> None:
    ts = pd.read_csv(OUT / "train_summary.csv")
    ppl = (ts.groupby(["model", "condition"])["final_perplexity"].mean().unstack())
    dppl = {m: ppl.loc[m, "B"] - ppl.loc[m, "A"] for m in ORDER}

    beh = pd.read_csv(OUT / "part2_results_per_model.csv").set_index("model")
    dbeh = {m: beh.loc[m, "mean_delta"] for m in ORDER}

    labels = [SHORT[m] for m in ORDER]
    y = range(len(ORDER))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax1.barh(y, [dppl[m] for m in ORDER], color="#1b9e77")
    ax1.set_yticks(list(y)); ax1.set_yticklabels(labels)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_title("Δ perplexitat (B − A)\n← B millor")
    ax1.set_xlabel("perplexitat morfèmica − nadiua")
    for i, m in enumerate(ORDER):
        ax1.text(dppl[m], i, f" {dppl[m]:+.2f} ",
                 va="center", ha="right" if dppl[m] < 0 else "left", fontsize=9)

    ax2.barh(y, [dbeh[m] for m in ORDER],
             color=["#d95f02" if dbeh[m] < -1e-9 else "#999999" for m in ORDER])
    ax2.set_yticks(list(y)); ax2.set_yticklabels([])
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_title("Δ precisió parells mínims (B − A)\nB millor →")
    ax2.set_xlabel("precisió morfèmica − nadiua")
    ax2.set_xlim(-0.45, 0.1)
    for i, m in enumerate(ORDER):
        ax2.text(dbeh[m], i, f" {dbeh[m]:+.2f} ",
                 va="center", ha="right" if dbeh[m] < 0 else "left", fontsize=9)

    fig.suptitle("Part 2: el guany de perplexitat NO es tradueix en comportament "
                 "(4 models · 25M tokens · 3 llavors)", fontsize=12)
    fig.tight_layout()
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figs" / "part2_dissociacio.png", dpi=150, bbox_inches="tight")
    print("wrote", OUT / "figs" / "part2_dissociacio.png")


if __name__ == "__main__":
    main()
