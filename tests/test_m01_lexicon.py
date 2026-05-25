from scripts.m01_build_lexicon import LEXICON, validate_entry, Entry


def test_every_curated_entry_is_valid():
    errors = []
    for e in LEXICON:
        errors += [f"{e.derived}: {msg}" for msg in validate_entry(e)]
    assert errors == [], errors


def test_gold_segmentation_must_join_to_derived():
    bad = Entry("ca", "ment", "ràpid", "ràpidament", "ràpida|men", "derivational",
                "He llegit la paraula {w} en un llibre.", "")
    assert any("does not join" in m for m in validate_entry(bad))


def test_ment_family_requires_ment_suffix():
    bad = Entry("ca", "ment", "ràpid", "ràpidor", "ràpid|or", "derivational",
                "He llegit la paraula {w} en un llibre.", "")
    assert any("suffix" in m for m in validate_entry(bad))


def test_carrier_has_w_slot():
    bad = Entry("ca", "ment", "ràpid", "ràpidament", "ràpida|ment", "derivational",
                "no slot here", "")
    assert any("{w}" in m for m in validate_entry(bad))


def test_lexicon_covers_all_families():
    fams = {e.family for e in LEXICON}
    expected = {
        "ment", "dim_et", "agent_dor", "nom_cio", "plural", "verb_em", "gender_a",
        "ly", "agent_er", "nom_tion", "plural_s",
        "gem_lla", "cedilla", "ny",
        "pre_des", "pre_re", "pre_in",
        "verb_reg", "verb_alt", "verb_supl", "nom_cio_d1",
        "es_mente", "es_cion", "es_dor", "es_dim", "es_plural", "es_genero_a",
    }
    assert expected <= fams
