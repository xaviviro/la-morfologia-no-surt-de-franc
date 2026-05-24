"""Catalan display labels for the figures.

The short family codes (`ment`, `plural`, …) stay as data identifiers in the
CSVs and docs; these maps give them readable Catalan labels for the plots."""

from __future__ import annotations

LANG_CA = {"ca": "català", "en": "anglès"}
COND_CA = {"native": "nadiu", "morphemic": "morfèmic", "delta": "delta"}
METRIC_CA = {
    "direction_consistency": "consistència de direcció",
    "analogy_acc": "precisió d'analogia",
    "pc1_var_ratio": "ràtio de variància PC1",
}
FAMILY_CA = {
    "ment": "-ment (adv)", "dim_et": "dim. -et", "agent_dor": "agent -dor",
    "nom_cio": "nom. -ció", "plural": "plural", "verb_em": "verb -em",
    "gender_a": "gènere -a", "ly": "-ly", "agent_er": "agent -er",
    "nom_tion": "nom. -tion", "plural_s": "plural -s",
    "gem_lla": "plural l·l", "cedilla": "plural ç", "ny": "plural ny",
}
GROUP_ANGLO = "Anglo-dominants"
GROUP_BSC = "BSC català-aware"
GROUP_DEEPSEEK = "DeepSeek"
GROUP_OTHER = "altres"


def short_model(model: str) -> str:
    return model.split("/")[-1]


def fam_label(code: str) -> str:
    return FAMILY_CA.get(code, code)
