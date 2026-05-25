"""Figures explicatives extra — les que fan llegible la història.

Afegeix a les de m05:
1. fertility_ratio        — ràtio de fertilitat CA/EN per tokenitzador
2. fertility_by_family    — fertilitat catalana mitjana per família morfològica
3. tokenization_examples  — COM cada tokenitzador talla paraules d'exemple
                            davant la frontera morfèmica real (la il·lustrativa)
4. delta_heatmap          — mapa de calor model×família del delta morfèmic-natiu
5. ment_summary           — el sufix protagonista -ment, natiu vs morfèmic

Tot el text dels gràfics és en català (els codis curts de família es mantenen)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.embed_lib import gold_boundaries  # noqa: E402
from scripts.geom_lib import bootstrap_mean_ci  # noqa: E402
from scripts.labels_ca import (  # noqa: E402
    COND_CA,
    GROUP_ANGLO,
    GROUP_BSC,
    GROUP_DEEPSEEK,
    GROUP_OTHER,
    METRIC_CA,
    fam_label,
    short_model,
)

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "out" / "figs"

ANGLO = {
    "google/gemma-2-2b", "google/gemma-4-E2B", "Qwen/Qwen2-1.5B",
    "Qwen/Qwen3.5-4B-Base", "google/gemma-4-26B-A4B",
    "mistralai/Mistral-Small-24B-Base-2501", "Qwen/Qwen3.6-35B-A3B",
}
BSC = {"BSC-LT/salamandra-2b", "BSC-LT/salamandra-7b", "BSC-LT/ALIA-40b"}
DEEPSEEK = {"deepseek-ai/deepseek-llm-67b-base"}


def _group(model: str) -> str:
    if model in ANGLO:
        return GROUP_ANGLO
    if model in BSC:
        return GROUP_BSC
    if model in DEEPSEEK:
        return GROUP_DEEPSEEK
    return GROUP_OTHER


def fig_fertility_ratio(audit: pd.DataFrame, path: Path) -> None:
    """Quants tokens més costa el català que l'anglès, per tokenitzador."""
    g = audit.groupby(["model", "lang"])["fertility"].mean().unstack("lang")
    g["ratio"] = g["ca"] / g["en"]
    g = g.sort_values("ratio")
    cmap = {GROUP_ANGLO: "#d95f02", GROUP_BSC: "#1b9e77", GROUP_DEEPSEEK: "#7570b3",
            GROUP_OTHER: "#7570b3"}
    colors = [cmap[_group(m)] for m in g.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh([short_model(m) for m in g.index], g["ratio"], color=colors)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("tokens del català per paraula ÷ tokens de l'anglès per paraula")
    ax.set_title("Quant més costa tokenitzar el català?")
    for bar, v in zip(bars, g["ratio"], strict=False):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.2f}×",
                va="center", fontsize=9)
    handles = [mpatches.Patch(color="#d95f02", label=GROUP_ANGLO),
               mpatches.Patch(color="#1b9e77", label=GROUP_BSC),
               mpatches.Patch(color="#7570b3", label=GROUP_DEEPSEEK)]
    ax.legend(handles=handles, loc="lower right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_fertility_by_family(audit: pd.DataFrame, path: Path) -> None:
    """Fertilitat catalana mitjana per família: anglo-dominants vs BSC."""
    ca = audit[audit.lang == "ca"].copy()
    ca["grp"] = ca["model"].map(_group)
    ca = ca[ca.grp.isin([GROUP_ANGLO, GROUP_BSC])]
    piv = ca.groupby(["family", "grp"])["fertility"].mean().unstack("grp")
    piv = piv.sort_values(GROUP_ANGLO, ascending=False)
    piv.index = [fam_label(c) for c in piv.index]
    ax = piv.plot(kind="bar", figsize=(12, 5),
                  color={GROUP_ANGLO: "#d95f02", GROUP_BSC: "#1b9e77"})
    ax.set_ylabel("subparaules per paraula (català, mitjana)")
    ax.set_xlabel("família morfològica")
    ax.set_title("Quina morfologia catalana es fragmenta més?")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="tokenitzador")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_tokenization_examples(examples: list[tuple[str, str]],
                              tokenizers: list[tuple[str, object]], path: Path) -> None:
    """Dibuixa, per cada (paraula, gold) i cada tokenitzador, els tokens com a
    caixes adjacents amb la frontera morfèmica real marcada en vermell."""
    n_rows = len(examples) * len(tokenizers)
    fig, ax = plt.subplots(figsize=(11, 0.9 * n_rows + 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)
    ax.axis("off")
    palette = ["#a6cee3", "#b2df8a", "#fdbf6f", "#cab2d6", "#fb9a99", "#ffff99"]
    y = n_rows
    for word, gold in examples:
        golds = set(gold_boundaries(gold))
        wlen = len(word)
        for label, tok in tokenizers:
            y -= 1
            enc = tok(word, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
            usable = max((e for (_s, e) in offsets), default=0) == wlen
            x0, x1 = 0.32, 0.97
            span = x1 - x0
            for k, (s, e) in enumerate(offsets):
                if e <= s:
                    continue
                left = x0 + span * (s / wlen)
                width = span * ((e - s) / wlen)
                ax.add_patch(mpatches.Rectangle((left, y + 0.15), width, 0.7,
                             facecolor=palette[k % len(palette)], edgecolor="black", lw=0.8))
                ax.text(left + width / 2, y + 0.5, word[s:e], ha="center", va="center",
                        fontsize=10)
            for b in golds:
                bx = x0 + span * (b / wlen)
                ax.plot([bx, bx], [y + 0.05, y + 0.95], color="red", lw=2.2, zorder=5)
            tag = "" if usable else "  (sense offsets)"
            ax.text(0.30, y + 0.5, f"{label}{tag}", ha="right", va="center", fontsize=9)
            ax.text(0.02, y + 0.5, word, ha="left", va="center", fontsize=11,
                    fontstyle="italic", color="#333")
    ax.plot([], [], color="red", lw=2.2, label="frontera morfèmica real")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0 + 0.8 / n_rows), ncol=1, fontsize=9)
    ax.set_title("Com els tokenitzadors tallen paraules catalanes  "
                 "(caixes = tokens, vermell = frontera morfèmica real)",
                 fontsize=12, pad=18)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_delta_heatmap(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    """Mapa de calor model×família del delta morfèmic-nadiu (vermell=pitjor, blau=millor).
    Un asterisc marca les cel·les significatives després de la correcció FDR (q<0,05)."""
    from scripts.m04_geometry import GEOMETRY_MODELS

    d = metrics[metrics.condition == "delta"].copy()
    rows = []
    for model in GEOMETRY_MODELS:
        L = GEOMETRY_MODELS[model][-1]
        rows.append(d[(d.model == model) & (d.layer == L)])
    d = pd.concat(rows, ignore_index=True) if rows else d
    ca_order = ["ment", "dim_et", "agent_dor", "nom_cio", "plural", "verb_em", "gender_a",
                "gem_lla", "cedilla", "ny"]
    en_order = ["ly", "agent_er", "nom_tion", "plural_s"]

    def _piv(col):
        p = d.pivot_table(index="model", columns="family", values=col, aggfunc="mean")
        keep = [c for c in ca_order + en_order if c in p.columns]
        return p[keep]

    piv = _piv(metric)
    cols = list(piv.columns)
    qp = _piv(f"{metric}_q") if f"{metric}_q" in d.columns else None
    ylabels = [short_model(m) for m in piv.index]
    vmax = float(np.nanmax(np.abs(piv.values))) or 1.0
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cols)), [fam_label(c) for c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.axvline(len(ca_order) - 0.5, color="black", lw=1.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            sig = ""
            if qp is not None:
                qij = qp.values[i, j]
                if not np.isnan(qij) and qij < 0.05:
                    sig = "*"
            ax.text(j, i, f"{v:+.2f}{sig}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label=f"Δ {METRIC_CA.get(metric, metric)} (morfèmic − nadiu)")
    ax.set_title(f"Ajuda la segmentació morfèmica?  Δ {METRIC_CA.get(metric, metric)}\n"
                 "blau = millor · esquerra de la línia = català · * = significatiu (FDR q<0,05)")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_placebo(agg: pd.DataFrame, metric: str, path: Path) -> None:
    """Control placebo: per família catalana, Δ(morfèmic − nadiu) vs
    Δ(morfèmic − aleatori) amb IC 95%. Si el segon és > 0, el guany és
    específic dels morfemes, no de trossejar diferent."""
    ca_order = ["ment", "dim_et", "agent_dor", "nom_cio", "plural", "verb_em",
                "gender_a", "gem_lla", "cedilla", "ny"]
    fam = agg[(agg.scope == "family") & (agg.metric == metric)]
    fams = [f for f in ca_order if f in set(fam.key)]
    y = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(10, 6))
    specs = [("delta", "morfèmic − nadiu", "#1b9e77", -0.16),
             ("delta_vs_random", "morfèmic − aleatori (placebo)", "#7570b3", +0.16)]
    for cond, label, color, off in specs:
        sub = fam[fam.condition == cond].set_index("key")
        means = np.array([sub.loc[f, "mean"] for f in fams])
        lo = np.array([sub.loc[f, "ci_lo"] for f in fams])
        hi = np.array([sub.loc[f, "ci_hi"] for f in fams])
        xerr = np.vstack([np.clip(means - lo, 0, None), np.clip(hi - means, 0, None)])
        ax.errorbar(means, y + off, xerr=xerr, fmt="o", color=color, ecolor=color,
                    capsize=3, ms=6, label=label)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(y, [fam_label(f) for f in fams])
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ {METRIC_CA.get(metric, metric)} (mitjana entre models, IC 95%)")
    ax.set_title("Control placebo: la segmentació morfèmica supera la segmentació aleatòria?")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_ment_delta_forest(metrics: pd.DataFrame, path: Path) -> None:
    """Forest plot del delta de «-ment» per model, amb IC 95% (bootstrap aparellat)."""
    from scripts.m04_geometry import GEOMETRY_MODELS

    recs = []
    for model, layers in GEOMETRY_MODELS.items():
        sub = metrics[(metrics.model == model) & (metrics.layer == layers[-1])
                      & (metrics.family == "ment") & (metrics.condition == "delta")]
        if len(sub):
            recs.append(sub.iloc[0])
    df = pd.DataFrame(recs)
    y = np.arange(len(df))
    ylabels = [short_model(m) for m in df["model"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    specs = [("direction_consistency", "Δ consistència de direcció"),
             ("analogy_acc", "Δ precisió d'analogia")]
    for ax, (metric, title) in zip(axes, specs, strict=False):
        val = df[metric].to_numpy(dtype=float)
        lo = df[f"{metric}_ci_lo"].to_numpy(dtype=float)
        hi = df[f"{metric}_ci_hi"].to_numpy(dtype=float)
        for k in range(len(df)):
            le = max(val[k] - lo[k], 0.0)
            ue = max(hi[k] - val[k], 0.0)
            sig = (lo[k] > 0) or (hi[k] < 0)
            color = "#1b9e77" if sig else "#999999"
            ax.errorbar(val[k], y[k], xerr=[[le], [ue]], fmt="o", color=color,
                        ecolor=color, capsize=4, ms=7)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_yticks(y, ylabels)
        ax.set_title(title)
        ax.invert_yaxis()
    fig.suptitle("«-ment»: delta morfèmic − nadiu amb IC 95%  (verd = l'IC exclou el 0)",
                 fontsize=13)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_ment_summary(metrics: pd.DataFrame, path: Path) -> None:
    """El sufix protagonista -ment: natiu vs morfèmic, les dues mètriques, per model."""
    from scripts.m04_geometry import GEOMETRY_MODELS

    rows = []
    for model, layers in GEOMETRY_MODELS.items():
        L = layers[-1]
        for cond in ("native", "morphemic"):
            sub = metrics[(metrics.model == model) & (metrics.layer == L)
                          & (metrics.family == "ment") & (metrics.condition == cond)]
            if len(sub):
                r = sub.iloc[0]
                rows.append({"model": short_model(model), "condició": COND_CA[cond],
                             "direction_consistency": r["direction_consistency"],
                             "analogy_acc": r["analogy_acc"]})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in [(axes[0], "direction_consistency"), (axes[1], "analogy_acc")]:
        piv = df.pivot(index="model", columns="condició", values=metric)[["nadiu", "morfèmic"]]
        piv.plot(kind="bar", ax=ax, color={"nadiu": "#bdbdbd", "morfèmic": "#1b9e77"})
        ax.set_title(METRIC_CA[metric])
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0, 1)
        ax.legend(title="condició")
    fig.suptitle("El sufix adverbial «-ment»: tokenització nadiua vs segmentació morfèmica oracle",
                 fontsize=13)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_layer_robustness(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    """Δ (morfèmic − nadiu) per capa, per model, amb banda d'IC 95% — mostra que
    el guany no depèn d'una sola tria de capa."""
    from scripts.m04_geometry import CA_FAMILIES, GEOMETRY_MODELS

    d = metrics[metrics.condition == "delta"].copy()
    d = d[d.family.isin(CA_FAMILIES)]
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, (model, layers) in enumerate(GEOMETRY_MODELS.items()):
        xs, ys, los, his = [], [], [], []
        for depth, L in enumerate(layers):
            sub = d[(d.model == model) & (d.layer == L)]
            if not len(sub):
                continue
            xs.append(depth)
            ys.append(sub[metric].mean())
            los.append(sub[f"{metric}_ci_lo"].mean())
            his.append(sub[f"{metric}_ci_hi"].mean())
        ys = np.array(ys)
        ax.plot(xs, ys, "-o", color=cmap(i), label=short_model(model))
        ax.fill_between(xs, los, his, color=cmap(i), alpha=0.15)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xticks([0, 1, 2], ["superficial (25%)", "mitjana (60%)", "profunda (90%)"])
    ax.set_xlabel("profunditat de capa")
    ax.set_ylabel(f"Δ {METRIC_CA.get(metric, metric)} (morfèmic − nadiu, mitjana famílies CA)")
    ax.set_title("Robustesa entre capes: la millora morfèmica es manté a tota la profunditat")
    ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_condition_ladder(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    """Escala de segmentació: consistència de direcció absoluta per condició
    (nadiu → Morfessor realista → aleatori → morfèmic oracle), mitjana sobre
    famílies CA i models, a la capa més profunda."""
    from scripts.m04_geometry import CA_FAMILIES, GEOMETRY_MODELS

    deep = {m: ls[-1] for m, ls in GEOMETRY_MODELS.items()}
    order = [("native", "nadiu"), ("random", "aleatori"),
             ("morfessor", "Morfessor (realista)"), ("morphemic", "morfèmic (oracle)")]
    means, los, his, labels = [], [], [], []
    for cond, label in order:
        vals = []
        for model in GEOMETRY_MODELS:
            sub = metrics[(metrics.model == model) & (metrics.layer == deep[model])
                          & (metrics.condition == cond) & (metrics.family.isin(CA_FAMILIES))]
            vals.extend(sub[metric].dropna().tolist())
        if not vals:
            continue
        m, lo, hi = bootstrap_mean_ci(np.array(vals), n=1000, seed=0)
        means.append(m)
        los.append(lo)
        his.append(hi)
        labels.append(label)
    x = np.arange(len(labels))
    colors = ["#bdbdbd", "#d95f02", "#7570b3", "#1b9e77"][: len(labels)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    err = np.vstack([np.array(means) - np.array(los), np.array(his) - np.array(means)])
    ax.bar(x, means, yerr=err, capsize=5, color=colors)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel(METRIC_CA.get(metric, metric))
    ax.set_title("Escala de segmentació: com de composicional és cada manera de tallar\n"
                 "(mitjana famílies CA × models, capa profunda, IC 95%)")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_morfessor_agreement(agree: pd.DataFrame, path: Path) -> None:
    """Recall de Morfessor vs les fronteres gold, per família (què recupera un
    segmentador realista no supervisat)."""
    g = agree.groupby("family")["recall"].mean()
    ca_order = ["ment", "dim_et", "agent_dor", "nom_cio", "plural", "verb_em",
                "gender_a", "gem_lla", "cedilla", "ny"]
    fams = [f for f in ca_order if f in g.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(fams)), [g[f] for f in fams], color="#7570b3")
    ax.axhline(g.mean(), color="black", ls="--", lw=1, label=f"mitjana = {g.mean():.2f}")
    ax.set_xticks(range(len(fams)), [fam_label(f) for f in fams], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall de frontera vs gold")
    ax.set_title("Què recupera Morfessor (segmentador realista no supervisat) per família")
    ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fig_regularity(analysis: pd.DataFrame, path: Path) -> None:
    """Consistència de direcció (flexió vs derivació) per condició de segmentació.
    La línia a 1,0 és la regularitat morfèmica perfecta; el 'sostre' real amb
    tall morfèmic (~0,6) mostra la desviació residual per arbitrarietat,
    UNIFORME entre flexió i derivació (no concentrada en la derivació)."""
    cond_ca = {"native": "nadiu", "morphemic": "morfèmic (oracle)",
               "morfessor": "Morfessor"}
    conds = [c for c in ("native", "morphemic", "morfessor") if c in set(analysis.condition)]
    x = np.arange(len(conds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mt, label, color, off in [
            ("inflectional", "flexió", "#1f77b4", -w / 2),
            ("derivational", "derivació", "#ff7f0e", +w / 2)]:
        means = []
        for c in conds:
            sub = analysis[(analysis.condition == c) & (analysis.morph_type == mt)]
            means.append(sub["consistency"].mean())
        ax.bar(x + off, means, w, label=label, color=color)
    ax.axhline(1.0, color="red", ls="--", lw=1, label="regularitat perfecta (=1,0)")
    ax.set_xticks(x, [cond_ca[c] for c in conds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("consistència de direcció")
    ax.set_title("Regularitat morfèmica a escala: flexió vs derivació\n"
                 "amb tall morfèmic convergeixen (~0,6), molt per sota de l'1,0 perfecte")
    ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    from transformers import AutoTokenizer

    audit = pd.read_csv(ROOT / "out" / "tokenize_audit.csv")
    metrics = pd.read_csv(ROOT / "out" / "geometry_metrics.csv")
    FIGS.mkdir(parents=True, exist_ok=True)

    fig_fertility_ratio(audit, FIGS / "fertility_ratio.png")
    fig_fertility_by_family(audit, FIGS / "fertility_by_family.png")
    fig_delta_heatmap(metrics, "direction_consistency", FIGS / "delta_heatmap_direction.png")
    fig_delta_heatmap(metrics, "analogy_acc", FIGS / "delta_heatmap_analogy.png")
    fig_ment_summary(metrics, FIGS / "ment_summary.png")
    fig_ment_delta_forest(metrics, FIGS / "ment_delta_forest.png")
    fig_layer_robustness(metrics, "direction_consistency", FIGS / "layer_robustness.png")
    if "morfessor" in set(metrics.condition):
        fig_condition_ladder(metrics, "direction_consistency", FIGS / "condition_ladder.png")

    agg_path = ROOT / "out" / "geometry_aggregate_ci.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        fig_placebo(agg, "direction_consistency", FIGS / "placebo_control.png")

    morf_path = ROOT / "out" / "morfessor_agreement.csv"
    if morf_path.exists():
        fig_morfessor_agreement(pd.read_csv(morf_path), FIGS / "morfessor_agreement.png")

    reg_path = ROOT / "out" / "regularity_analysis.csv"
    if reg_path.exists():
        fig_regularity(pd.read_csv(reg_path), FIGS / "regularity.png")

    examples = [
        ("ràpidament", "ràpida|ment"),
        ("educació", "educa|ció"),
        ("gatet", "gat|et"),
        ("cantem", "cant|em"),
    ]
    toks = [
        ("gemma-2-2b (anglo)", AutoTokenizer.from_pretrained("google/gemma-2-2b", use_fast=True)),
        ("salamandra-2b (BSC)", AutoTokenizer.from_pretrained("BSC-LT/salamandra-2b", use_fast=True)),
    ]
    fig_tokenization_examples(examples, toks, FIGS / "tokenization_examples.png")

    ortho = [
        ("col·legis", "col·legi|s"),
        ("paral·lels", "paral·lel|s"),
        ("caçador", "caça|dor"),
        ("feliços", "feliç|os"),
        ("muntanyes", "muntany|es"),
        ("anys", "any|s"),
    ]
    fig_tokenization_examples(ortho, toks, FIGS / "tokenization_examples_ortho.png")
    print(f"wrote extra figures to {FIGS}")


if __name__ == "__main__":
    main()
