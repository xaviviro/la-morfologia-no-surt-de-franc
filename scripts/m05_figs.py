"""Figures de l'espai vectorial de l'estudi de tokenització-morfologia.

Cinc figures: fertilitat CA vs EN, mapa de calor de recall de frontera, el
mapa de la direcció -ment (natiu vs morfèmic, PCA), barres de direcció/analogia
(natiu vs morfèmic + baseline EN), i el scatter tokenització→geometria.
Tot el text dels gràfics és en català (els codis curts de família es mantenen)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from scripts.embed_lib import slugify  # noqa: E402
from scripts.labels_ca import (  # noqa: E402
    COND_CA,
    LANG_CA,
    METRIC_CA,
    fam_label,
    short_model,
)

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "out" / "figs"


def fig_fertility(audit: pd.DataFrame, path: Path) -> None:
    g = audit.groupby(["model", "lang"])["fertility"].mean().unstack("lang")
    g = g.rename(columns=LANG_CA)
    g.index = [short_model(m) for m in g.index]
    ax = g.plot(kind="bar", figsize=(12, 5), color={"català": "#1f77b4", "anglès": "#ff7f0e"})
    ax.set_ylabel("subparaules per paraula (mitjana)")
    ax.set_xlabel("model")
    ax.set_title("Fertilitat del tokenitzador: català vs anglès")
    ax.legend(title="llengua")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_boundary_heatmap(audit: pd.DataFrame, path: Path) -> None:
    ca = audit[audit.lang == "ca"]
    piv = ca.pivot_table(index="model", columns="family", values="boundary_recall", aggfunc="mean")
    piv = piv.dropna(how="all")  # treu tokenitzadors sense offsets útils (p. ex. DeepSeek)
    cols = [fam_label(c) for c in piv.columns]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(piv.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)), [short_model(m) for m in piv.index])
    ax.set_title("Recall de frontera morfèmica (català)")
    fig.colorbar(im, ax=ax, label="recall (proporció de fronteres encertades)")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_direction_map(meta: pd.DataFrame, vecs_native, vecs_morph, family: str, path: Path) -> None:
    """Scatter PCA dels parells base→derivat amb fletxes, natiu vs morfèmic."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, vecs, title in [(axes[0], vecs_native, "nadiu"), (axes[1], vecs_morph, "morfèmic")]:
        sel = meta[(meta.family == family)]
        b = vecs[sel[sel.role == "base"].index.to_numpy()]
        d = vecs[sel[sel.role == "derived"].index.to_numpy()]
        pca = PCA(n_components=2).fit(np.vstack([b, d]))
        b2, d2 = pca.transform(b), pca.transform(d)
        ax.scatter(b2[:, 0], b2[:, 1], c="tab:blue", label="base", s=20)
        ax.scatter(d2[:, 0], d2[:, 1], c="tab:red", label="derivat", s=20)
        for i in range(len(b2)):
            ax.annotate("", xy=d2[i], xytext=b2[i],
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))
        ax.set_title(f"direcció «{fam_label(family)}» — {title}")
        ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_metric_bars(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    df = metrics[metrics.condition.isin(["native", "morphemic"])].copy()
    df["model"] = df["model"].map(short_model)
    df["family"] = df["family"].map(fam_label)
    piv = df.pivot_table(index=["model", "family"], columns="condition", values=metric)
    piv = piv.rename(columns=COND_CA)[["nadiu", "morfèmic"]]
    ax = piv.plot(kind="bar", figsize=(17, 6), color={"nadiu": "#bdbdbd", "morfèmic": "#1b9e77"})
    ax.set_ylabel(METRIC_CA.get(metric, metric))
    ax.set_xlabel("model · família")
    ax.set_title(f"{METRIC_CA.get(metric, metric)}: nadiu vs morfèmic")
    ax.legend(title="condició")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_tok_vs_geom(audit: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    """x = recall de frontera (CA), y = consistència de direcció nativa, per model×família."""
    rec = (audit[audit.lang == "ca"].groupby(["model", "family"])["boundary_recall"]
           .mean().reset_index())
    geo = metrics[(metrics.condition == "native")][["model", "family", "direction_consistency"]]
    m = rec.merge(geo, on=["model", "family"], how="inner")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(m.boundary_recall, m.direction_consistency, s=30, alpha=0.7)
    ax.set_xlabel("recall de frontera morfèmica (tokenitzador nadiu)")
    ax.set_ylabel("consistència de direcció (geometria nadiua)")
    ax.set_title("Fragmentació de la tokenització → degradació geomètrica")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    audit = pd.read_csv(ROOT / "out" / "tokenize_audit.csv")
    metrics = pd.read_csv(ROOT / "out" / "geometry_metrics.csv")
    FIGS.mkdir(parents=True, exist_ok=True)

    fig_fertility(audit, FIGS / "fertility_ca_vs_en.png")
    fig_boundary_heatmap(audit, FIGS / "boundary_recall_heatmap.png")
    fig_metric_bars(metrics, "direction_consistency", FIGS / "direction_consistency_bars.png")
    fig_metric_bars(metrics, "analogy_acc", FIGS / "analogy_accuracy_bars.png")
    fig_tok_vs_geom(audit, metrics, FIGS / "tok_vs_geom_scatter.png")

    # mapa de la direcció -ment per a cada model de geometria
    for model_id in ["google/gemma-2-2b", "google/gemma-4-E2B", "Qwen/Qwen2-1.5B",
                     "Qwen/Qwen3.5-4B-Base", "BSC-LT/salamandra-2b"]:
        mdir = ROOT / "out" / slugify(model_id)
        meta_path = mdir / "metadata.parquet"
        if not meta_path.exists():
            continue
        meta = pd.read_parquet(meta_path)
        from scripts.m04_geometry import GEOMETRY_MODELS
        best_L = GEOMETRY_MODELS[model_id][-1]
        vecs = np.load(mdir / f"embeddings_layer{best_L}.npz")["vectors"]
        nat = meta[meta.condition == "native"].reset_index(drop=True)
        vnat = vecs[meta[meta.condition == "native"].index.to_numpy()]
        vmor = vecs[meta[meta.condition == "morphemic"].index.to_numpy()]
        fig_direction_map(nat, vnat, vmor, "ment", FIGS / f"ment_direction_{slugify(model_id)}.png")
    print(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
