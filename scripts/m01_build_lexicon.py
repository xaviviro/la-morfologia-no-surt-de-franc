"""Curated Catalan (+ English baseline) morphology lexicon with gold morpheme
boundaries. Run to emit data/morph_pairs.csv. Author-reviewed for correctness
(accents, the -ment feminine-stem rule, irregular forms)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CARRIER_CA = "He llegit la paraula {w} en un llibre."
CARRIER_EN = "I read the word {w} in a book."

# Suffix string that each family's gold segmentation must end with.
FAMILY_SUFFIX = {
    "ment": "ment", "dim_et": None, "agent_dor": "dor", "nom_cio": "ció",
    "plural": None, "verb_em": "em", "gender_a": "a",
    "ly": "ly", "agent_er": "er", "nom_tion": "tion", "plural_s": "s",
    # Catalan-specific orthography (plural morphology, feature in the stem):
    "gem_lla": None, "cedilla": None, "ny": None,
    # IE-motivated patterns: prefixation, Sturtevant regularity gradient, depth:
    "pre_des": None, "pre_re": None, "pre_in": None,
    "verb_reg": None, "verb_alt": None, "verb_supl": None,
    "nom_cio_d1": "ció",
}


@dataclass(frozen=True)
class Entry:
    lang: str
    family: str
    base: str
    derived: str
    gold_segmentation: str
    morph_type: str
    carrier: str
    note: str = ""


def _ca(family, morph_type, rows):
    return [Entry("ca", family, b, d, g, morph_type, CARRIER_CA) for (b, d, g) in rows]


def _en(family, morph_type, rows):
    return [Entry("en", family, b, d, g, morph_type, CARRIER_EN) for (b, d, g) in rows]


# --- Catalan derivational ---------------------------------------------------
MENT = _ca("ment", "derivational", [
    ("ràpid", "ràpidament", "ràpida|ment"),
    ("lent", "lentament", "lenta|ment"),
    ("clar", "clarament", "clara|ment"),
    ("fàcil", "fàcilment", "fàcil|ment"),
    ("difícil", "difícilment", "difícil|ment"),
    ("fort", "fortament", "forta|ment"),
    ("dolç", "dolçament", "dolça|ment"),
    ("trist", "tristament", "trista|ment"),
    ("net", "netament", "neta|ment"),
    ("segur", "segurament", "segura|ment"),
    ("precís", "precisament", "precisa|ment"),
    ("normal", "normalment", "normal|ment"),
    ("general", "generalment", "general|ment"),
    ("públic", "públicament", "pública|ment"),
    ("bàsic", "bàsicament", "bàsica|ment"),
    ("directe", "directament", "directa|ment"),
    ("profund", "profundament", "profunda|ment"),
    ("intens", "intensament", "intensa|ment"),
    ("exacte", "exactament", "exacta|ment"),
    ("perfecte", "perfectament", "perfecta|ment"),
    ("absolut", "absolutament", "absoluta|ment"),
    ("relatiu", "relativament", "relativa|ment"),
    ("seriós", "seriosament", "seriosa|ment"),
    ("nerviós", "nerviosament", "nerviosa|ment"),
    ("lliure", "lliurement", "lliure|ment"),
    ("alegre", "alegrement", "alegre|ment"),
    ("simple", "simplement", "simple|ment"),
    ("possible", "possiblement", "possible|ment"),
    ("evident", "evidentment", "evident|ment"),
    ("recent", "recentment", "recent|ment"),
    ("urgent", "urgentment", "urgent|ment"),
    ("atent", "atentament", "atenta|ment"),
    ("fred", "fredament", "freda|ment"),
    ("antic", "antigament", "antiga|ment"),
    ("llarg", "llargament", "llarga|ment"),
    ("curt", "curtament", "curta|ment"),
    ("ric", "ricament", "rica|ment"),
    ("nou", "novament", "nova|ment"),
    ("greu", "greument", "greu|ment"),
    ("ample", "amplament", "ampla|ment"),
])

DIM_ET = _ca("dim_et", "derivational", [
    ("gat", "gatet", "gat|et"),
    ("casa", "caseta", "cas|eta"),
    ("finestra", "finestreta", "finestr|eta"),
    ("taula", "tauleta", "taul|eta"),
    ("llibre", "llibret", "llibr|et"),
    ("arbre", "arbret", "arbr|et"),
    ("got", "gotet", "got|et"),
    ("nen", "nenet", "nen|et"),
    ("ocell", "ocellet", "ocell|et"),
    ("flor", "floreta", "flor|eta"),
    ("porta", "porteta", "port|eta"),
    ("peu", "peuet", "peu|et"),
    ("cotxe", "cotxet", "cotx|et"),
    ("llit", "llitet", "llit|et"),
    ("pont", "pontet", "pont|et"),
    ("dit", "ditet", "dit|et"),
    ("ull", "ullet", "ull|et"),
    ("camí", "caminet", "camin|et"),
    ("jardí", "jardinet", "jardin|et"),
    ("moment", "momentet", "moment|et"),
    ("tren", "trenet", "tren|et"),
    ("paper", "paperet", "paper|et"),
    ("plat", "platet", "plat|et"),
    ("pis", "piset", "pis|et"),
    ("ram", "ramet", "ram|et"),
    ("fil", "filet", "fil|et"),
    ("cadira", "cadireta", "cadir|eta"),
    ("hora", "horeta", "hor|eta"),
    ("camisa", "camiseta", "camis|eta"),
    ("cullera", "cullereta", "culler|eta"),
])

AGENT_DOR = _ca("agent_dor", "derivational", [
    ("treballar", "treballador", "treballa|dor"),
    ("jugar", "jugador", "juga|dor"),
    ("pescar", "pescador", "pesca|dor"),
    ("comprar", "comprador", "compra|dor"),
    ("ballar", "ballador", "balla|dor"),
    ("nedar", "nedador", "neda|dor"),
    ("educar", "educador", "educa|dor"),
    ("programar", "programador", "programa|dor"),
    ("administrar", "administrador", "administra|dor"),
    ("governar", "governador", "governa|dor"),
    ("caçar", "caçador", "caça|dor"),
    ("crear", "creador", "crea|dor"),
    ("comptar", "comptador", "compta|dor"),
    ("explorar", "explorador", "explora|dor"),
    ("investigar", "investigador", "investiga|dor"),
    ("organitzar", "organitzador", "organitza|dor"),
    ("dissenyar", "dissenyador", "dissenya|dor"),
    ("comunicar", "comunicador", "comunica|dor"),
    ("presentar", "presentador", "presenta|dor"),
    ("entrenar", "entrenador", "entrena|dor"),
    ("ordenar", "ordenador", "ordena|dor"),
    ("gravar", "gravador", "grava|dor"),
    ("regar", "regador", "rega|dor"),
    ("comprovar", "comprovador", "comprova|dor"),
    ("exportar", "exportador", "exporta|dor"),
])

NOM_CIO = _ca("nom_cio", "derivational", [
    ("educar", "educació", "educa|ció"),
    ("informar", "informació", "informa|ció"),
    ("crear", "creació", "crea|ció"),
    ("formar", "formació", "forma|ció"),
    ("comunicar", "comunicació", "comunica|ció"),
    ("organitzar", "organització", "organitza|ció"),
    ("administrar", "administració", "administra|ció"),
    ("presentar", "presentació", "presenta|ció"),
    ("preparar", "preparació", "prepara|ció"),
    ("situar", "situació", "situa|ció"),
    ("continuar", "continuació", "continua|ció"),
    ("actuar", "actuació", "actua|ció"),
    ("aplicar", "aplicació", "aplica|ció"),
    ("associar", "associació", "associa|ció"),
    ("celebrar", "celebració", "celebra|ció"),
    ("circular", "circulació", "circula|ció"),
    ("declarar", "declaració", "declara|ció"),
    ("generar", "generació", "genera|ció"),
    ("importar", "importació", "importa|ció"),
    ("exportar", "exportació", "exporta|ció"),
    ("modificar", "modificació", "modifica|ció"),
    ("negociar", "negociació", "negocia|ció"),
    ("observar", "observació", "observa|ció"),
    ("participar", "participació", "participa|ció"),
])

# --- Catalan inflectional ---------------------------------------------------
PLURAL = _ca("plural", "inflectional", [
    ("gat", "gats", "gat|s"),
    ("llibre", "llibres", "llibre|s"),
    ("cotxe", "cotxes", "cotxe|s"),
    ("arbre", "arbres", "arbre|s"),
    ("color", "colors", "color|s"),
    ("flor", "flors", "flor|s"),
    ("ocell", "ocells", "ocell|s"),
    ("carrer", "carrers", "carrer|s"),
    ("pont", "ponts", "pont|s"),
    ("peix", "peixos", "peix|os"),
    ("calaix", "calaixos", "calaix|os"),
    ("despatx", "despatxos", "despatx|os"),
    ("text", "textos", "text|os"),
    ("gas", "gasos", "gas|os"),
    ("cel", "cels", "cel|s"),
    ("tren", "trens", "tren|s"),
    ("llum", "llums", "llum|s"),
    ("jardí", "jardins", "jardin|s"),
    ("riu", "rius", "riu|s"),
    ("peu", "peus", "peu|s"),
    ("museu", "museus", "museu|s"),
    ("taxi", "taxis", "taxi|s"),
    ("dit", "dits", "dit|s"),
    ("plat", "plats", "plat|s"),
    ("moment", "moments", "moment|s"),
    ("paper", "papers", "paper|s"),
    ("pis", "pisos", "pis|os"),
    ("braç", "braços", "braç|os"),
    ("disc", "discos", "disc|os"),
    ("bosc", "boscos", "bosc|os"),
])

VERB_EM = _ca("verb_em", "inflectional", [
    ("cantar", "cantem", "cant|em"),
    ("parlar", "parlem", "parl|em"),
    ("treballar", "treballem", "treball|em"),
    ("mirar", "mirem", "mir|em"),
    ("estudiar", "estudiem", "estudi|em"),
    ("comprar", "comprem", "compr|em"),
    ("ballar", "ballem", "ball|em"),
    ("nedar", "nedem", "ned|em"),
    ("caminar", "caminem", "camin|em"),
    ("escoltar", "escoltem", "escolt|em"),
    ("cuinar", "cuinem", "cuin|em"),
    ("dibuixar", "dibuixem", "dibuix|em"),
    ("saltar", "saltem", "salt|em"),
    ("entrar", "entrem", "entr|em"),
    ("portar", "portem", "port|em"),
    ("trobar", "trobem", "trob|em"),
    ("ajudar", "ajudem", "ajud|em"),
    ("preguntar", "preguntem", "pregunt|em"),
    ("contestar", "contestem", "contest|em"),
    ("visitar", "visitem", "visit|em"),
    ("dinar", "dinem", "din|em"),
    ("sopar", "sopem", "sop|em"),
    ("arribar", "arribem", "arrib|em"),
    ("guanyar", "guanyem", "guany|em"),
    ("pensar", "pensem", "pens|em"),
])

GENDER_A = _ca("gender_a", "inflectional", [
    ("gat", "gata", "gat|a"),
    ("noi", "noia", "noi|a"),
    ("nen", "nena", "nen|a"),
    ("fill", "filla", "fill|a"),
    ("professor", "professora", "professor|a"),
    ("senyor", "senyora", "senyor|a"),
    ("doctor", "doctora", "doctor|a"),
    ("alumne", "alumna", "alumn|a"),
    ("cosí", "cosina", "cosin|a"),
    ("tiet", "tieta", "tiet|a"),
    ("mestre", "mestra", "mestr|a"),
    ("sogre", "sogra", "sogr|a"),
    ("pintor", "pintora", "pintor|a"),
    ("escriptor", "escriptora", "escriptor|a"),
    ("director", "directora", "director|a"),
    ("treballador", "treballadora", "treballador|a"),
    ("jugador", "jugadora", "jugador|a"),
    ("monitor", "monitora", "monitor|a"),
    ("germà", "germana", "german|a"),
    ("company", "companya", "company|a"),
    ("rei", "reina", "rein|a"),
    ("xicot", "xicota", "xicot|a"),
    ("candidat", "candidata", "candidat|a"),
    ("conill", "conilla", "conill|a"),
    ("enginyer", "enginyera", "enginyer|a"),
])

# --- English parallel baseline ----------------------------------------------
LY = _en("ly", "derivational", [
    ("quick", "quickly", "quick|ly"),
    ("slow", "slowly", "slow|ly"),
    ("clear", "clearly", "clear|ly"),
    ("easy", "easily", "easi|ly"),
    ("happy", "happily", "happi|ly"),
    ("rapid", "rapidly", "rapid|ly"),
    ("soft", "softly", "soft|ly"),
    ("strong", "strongly", "strong|ly"),
    ("perfect", "perfectly", "perfect|ly"),
    ("absolute", "absolutely", "absolute|ly"),
    ("exact", "exactly", "exact|ly"),
    ("direct", "directly", "direct|ly"),
    ("serious", "seriously", "serious|ly"),
    ("nervous", "nervously", "nervous|ly"),
    ("general", "generally", "general|ly"),
    ("normal", "normally", "normal|ly"),
    ("deep", "deeply", "deep|ly"),
    ("brief", "briefly", "brief|ly"),
    ("strange", "strangely", "strange|ly"),
    ("real", "really", "real|ly"),
    ("careful", "carefully", "careful|ly"),
    ("quiet", "quietly", "quiet|ly"),
    ("bright", "brightly", "bright|ly"),
    ("calm", "calmly", "calm|ly"),
    ("final", "finally", "final|ly"),
    ("usual", "usually", "usual|ly"),
    ("actual", "actually", "actual|ly"),
    ("complete", "completely", "complete|ly"),
    ("safe", "safely", "safe|ly"),
    ("nice", "nicely", "nice|ly"),
    ("close", "closely", "close|ly"),
    ("wide", "widely", "wide|ly"),
    ("rare", "rarely", "rare|ly"),
    ("late", "lately", "late|ly"),
    ("free", "freely", "free|ly"),
    ("loud", "loudly", "loud|ly"),
    ("sad", "sadly", "sad|ly"),
    ("kind", "kindly", "kind|ly"),
    ("warm", "warmly", "warm|ly"),
    ("cold", "coldly", "cold|ly"),
])

AGENT_ER = _en("agent_er", "derivational", [
    ("teach", "teacher", "teach|er"),
    ("play", "player", "play|er"),
    ("work", "worker", "work|er"),
    ("sing", "singer", "sing|er"),
    ("write", "writer", "writ|er"),
    ("read", "reader", "read|er"),
    ("drive", "driver", "driv|er"),
    ("paint", "painter", "paint|er"),
    ("build", "builder", "build|er"),
    ("dance", "dancer", "danc|er"),
    ("farm", "farmer", "farm|er"),
    ("bake", "baker", "bak|er"),
    ("clean", "cleaner", "clean|er"),
    ("help", "helper", "help|er"),
    ("speak", "speaker", "speak|er"),
    ("listen", "listener", "listen|er"),
    ("report", "reporter", "report|er"),
    ("manage", "manager", "manag|er"),
    ("own", "owner", "own|er"),
    ("lead", "leader", "lead|er"),
    ("found", "founder", "found|er"),
    ("design", "designer", "design|er"),
    ("print", "printer", "print|er"),
    ("record", "recorder", "record|er"),
    ("train", "trainer", "train|er"),
])

NOM_TION = _en("nom_tion", "derivational", [
    ("educate", "education", "educa|tion"),
    ("inform", "information", "inform|ation"),
    ("create", "creation", "crea|tion"),
    ("relate", "relation", "rela|tion"),
    ("form", "formation", "form|ation"),
    ("communicate", "communication", "communica|tion"),
    ("organize", "organization", "organiza|tion"),
    ("present", "presentation", "present|ation"),
    ("prepare", "preparation", "prepara|tion"),
    ("situate", "situation", "situa|tion"),
    ("locate", "location", "loca|tion"),
    ("operate", "operation", "opera|tion"),
    ("produce", "production", "produc|tion"),
    ("protect", "protection", "protec|tion"),
    ("collect", "collection", "collec|tion"),
    ("connect", "connection", "connec|tion"),
    ("select", "selection", "selec|tion"),
    ("direct", "direction", "direc|tion"),
    ("introduce", "introduction", "introduc|tion"),
    ("reduce", "reduction", "reduc|tion"),
    ("translate", "translation", "transla|tion"),
    ("celebrate", "celebration", "celebra|tion"),
    ("decorate", "decoration", "decora|tion"),
    ("imagine", "imagination", "imagina|tion"),
    ("examine", "examination", "examina|tion"),
])

PLURAL_S = _en("plural_s", "inflectional", [
    ("cat", "cats", "cat|s"),
    ("book", "books", "book|s"),
    ("tree", "trees", "tree|s"),
    ("dog", "dogs", "dog|s"),
    ("car", "cars", "car|s"),
    ("color", "colors", "color|s"),
    ("flower", "flowers", "flower|s"),
    ("table", "tables", "table|s"),
    ("door", "doors", "door|s"),
    ("bird", "birds", "bird|s"),
    ("river", "rivers", "river|s"),
    ("road", "roads", "road|s"),
    ("window", "windows", "window|s"),
    ("hand", "hands", "hand|s"),
    ("house", "houses", "house|s"),
    ("chair", "chairs", "chair|s"),
    ("phone", "phones", "phone|s"),
    ("school", "schools", "school|s"),
    ("friend", "friends", "friend|s"),
    ("garden", "gardens", "garden|s"),
    ("street", "streets", "street|s"),
    ("key", "keys", "key|s"),
    ("day", "days", "day|s"),
    ("year", "years", "year|s"),
    ("hour", "hours", "hour|s"),
    ("room", "rooms", "room|s"),
    ("word", "words", "word|s"),
    ("name", "names", "name|s"),
    ("game", "games", "game|s"),
    ("eye", "eyes", "eye|s"),
])

# --- Catalan orthography stress-tests (plural morphology; the orthographic
#     feature — ela geminada l·l, ç, or the ny digraph — lives in the stem) ---
GEM_LLA = _ca("gem_lla", "inflectional", [
    ("col·legi", "col·legis", "col·legi|s"),
    ("paral·lel", "paral·lels", "paral·lel|s"),
    ("intel·lectual", "intel·lectuals", "intel·lectual|s"),
    ("mil·lenni", "mil·lennis", "mil·lenni|s"),
    ("excel·lent", "excel·lents", "excel·lent|s"),
    ("el·lipse", "el·lipses", "el·lipse|s"),
    ("novel·la", "novel·les", "novel·l|es"),
    ("cèl·lula", "cèl·lules", "cèl·lul|es"),
    ("síl·laba", "síl·labes", "síl·lab|es"),
    ("aquarel·la", "aquarel·les", "aquarel·l|es"),
    ("al·lèrgia", "al·lèrgies", "al·lèrgi|es"),
    ("pel·lícula", "pel·lícules", "pel·lícul|es"),
])

CEDILLA = _ca("cedilla", "inflectional", [
    ("plaça", "places", "plac|es"),
    ("caça", "caces", "cac|es"),
    ("força", "forces", "forc|es"),
    ("peça", "peces", "pec|es"),
    ("llança", "llances", "llanc|es"),
    ("esperança", "esperances", "esperanc|es"),
    ("confiança", "confiances", "confianc|es"),
    ("feliç", "feliços", "feliç|os"),
    ("dolç", "dolços", "dolç|os"),
    ("audaç", "audaços", "audaç|os"),
    ("capaç", "capaços", "capaç|os"),
    ("esforç", "esforços", "esforç|os"),
    ("veloç", "veloços", "veloç|os"),
])

NY = _ca("ny", "inflectional", [
    ("any", "anys", "any|s"),
    ("puny", "punys", "puny|s"),
    ("estany", "estanys", "estany|s"),
    ("pany", "panys", "pany|s"),
    ("bany", "banys", "bany|s"),
    ("guany", "guanys", "guany|s"),
    ("muntanya", "muntanyes", "muntany|es"),
    ("canya", "canyes", "cany|es"),
    ("vinya", "vinyes", "viny|es"),
    ("llenya", "llenyes", "lleny|es"),
    ("companyia", "companyies", "companyi|es"),
    ("cabanya", "cabanyes", "cabany|es"),
])

# --- IE pattern A: prefixation (the study is otherwise all-suffixal) ----------
PRE_DES = _ca("pre_des", "derivational", [
    ("fer", "desfer", "des|fer"),
    ("muntar", "desmuntar", "des|muntar"),
    ("lligar", "deslligar", "des|lligar"),
    ("ordenar", "desordenar", "des|ordenar"),
    ("tapar", "destapar", "des|tapar"),
    ("congelar", "descongelar", "des|congelar"),
    ("connectar", "desconnectar", "des|connectar"),
    ("activar", "desactivar", "des|activar"),
    ("carregar", "descarregar", "des|carregar"),
    ("muntatge", "desmuntatge", "des|muntatge"),
    ("avantatge", "desavantatge", "des|avantatge"),
    ("acord", "desacord", "des|acord"),
])

PRE_RE = _ca("pre_re", "derivational", [
    ("fer", "refer", "re|fer"),
    ("llegir", "rellegir", "re|llegir"),
    ("escriure", "reescriure", "re|escriure"),
    ("començar", "recomençar", "re|començar"),
    ("omplir", "reomplir", "re|omplir"),
    ("considerar", "reconsiderar", "re|considerar"),
    ("organitzar", "reorganitzar", "re|organitzar"),
    ("construir", "reconstruir", "re|construir"),
    ("col·locar", "recol·locar", "re|col·locar"),
    ("aparèixer", "reaparèixer", "re|aparèixer"),
    ("plantejar", "replantejar", "re|plantejar"),
    ("definir", "redefinir", "re|definir"),
])

PRE_IN = _ca("pre_in", "derivational", [
    ("útil", "inútil", "in|útil"),
    ("just", "injust", "in|just"),
    ("complet", "incomplet", "in|complet"),
    ("correcte", "incorrecte", "in|correcte"),
    ("possible", "impossible", "im|possible"),
    ("mortal", "immortal", "im|mortal"),
    ("visible", "invisible", "in|visible"),
    ("capaç", "incapaç", "in|capaç"),
    ("necessari", "innecessari", "in|necessari"),
    ("segur", "insegur", "in|segur"),
    ("actiu", "inactiu", "in|actiu"),
    ("oportú", "inoportú", "in|oportú"),
])

# --- IE pattern B/C: Sturtevant regularity gradient (verbs, 1sg present) -------
#     regular (stem intact) -> stem alternation (ablaut/velar) -> suppletive.
#     The alternating/suppletive forms have no clean morpheme cut, so their gold
#     is the whole word (oracle == native); the test of interest is the *native*
#     direction consistency, which should decay along the gradient.
VERB_REG = _ca("verb_reg", "inflectional", [
    ("cantar", "canto", "cant|o"),
    ("parlar", "parlo", "parl|o"),
    ("mirar", "miro", "mir|o"),
    ("comprar", "compro", "compr|o"),
    ("treballar", "treballo", "treball|o"),
    ("estudiar", "estudio", "estudi|o"),
    ("escoltar", "escolto", "escolt|o"),
    ("ballar", "ballo", "ball|o"),
    ("nedar", "nedo", "ned|o"),
    ("caminar", "camino", "camin|o"),
    ("cuinar", "cuino", "cuin|o"),
    ("saltar", "salto", "salt|o"),
    ("entrar", "entro", "entr|o"),
    ("portar", "porto", "port|o"),
    ("ajudar", "ajudo", "ajud|o"),
])

VERB_ALT = _ca("verb_alt", "inflectional", [
    ("poder", "puc", "puc"),
    ("voler", "vull", "vull"),
    ("tenir", "tinc", "tinc"),
    ("venir", "vinc", "vinc"),
    ("dir", "dic", "dic"),
    ("dur", "duc", "duc"),
    ("beure", "bec", "bec"),
    ("caure", "caic", "caic"),
    ("creure", "crec", "crec"),
    ("viure", "visc", "visc"),
    ("prendre", "prenc", "prenc"),
    ("escriure", "escric", "escric"),
    ("collir", "cullo", "cullo"),
    ("cosir", "cuso", "cuso"),
    ("sortir", "surto", "surto"),
])

VERB_SUPL = _ca("verb_supl", "inflectional", [
    # Orthography note: we keep the traditional accented "sóc". The 2016 IEC
    # reform writes it "soc" (the diacritic was dropped). We retain "sóc"
    # because it remains widespread and because "soc" collides with the noun
    # "soc" (clog/market); since the accent changes the token sequence, the
    # choice is recorded explicitly here and in docs/methodology.md §2.
    ("ser", "sóc", "sóc"),
    ("anar", "vaig", "vaig"),
    ("fer", "faig", "faig"),
    ("estar", "estic", "estic"),
    ("haver", "he", "he"),
    ("saber", "sé", "sé"),
    ("veure", "veig", "veig"),
])

# --- IE pattern D: derivational depth (same -ció suffix on a derived base) -----
NOM_CIO_D1 = _ca("nom_cio_d1", "derivational", [
    ("realitzar", "realització", "realitza|ció"),
    ("globalitzar", "globalització", "globalitza|ció"),
    ("organitzar", "organització", "organitza|ció"),
    ("modernitzar", "modernització", "modernitza|ció"),
    ("nacionalitzar", "nacionalització", "nacionalitza|ció"),
    ("privatitzar", "privatització", "privatitza|ció"),
    ("actualitzar", "actualització", "actualitza|ció"),
    ("legalitzar", "legalització", "legalitza|ció"),
    ("normalitzar", "normalització", "normalitza|ció"),
    ("centralitzar", "centralització", "centralitza|ció"),
    ("automatitzar", "automatització", "automatitza|ció"),
    ("digitalitzar", "digitalització", "digitalitza|ció"),
])

LEXICON: list[Entry] = (
    MENT + DIM_ET + AGENT_DOR + NOM_CIO + PLURAL + VERB_EM + GENDER_A
    + LY + AGENT_ER + NOM_TION + PLURAL_S
    + GEM_LLA + CEDILLA + NY
    + PRE_DES + PRE_RE + PRE_IN
    + VERB_REG + VERB_ALT + VERB_SUPL
    + NOM_CIO_D1
)


def validate_entry(e: Entry) -> list[str]:
    msgs: list[str] = []
    joined = e.gold_segmentation.replace("|", "")
    if joined != e.derived:
        msgs.append(f"gold_segmentation {e.gold_segmentation!r} does not join to {e.derived!r}")
    suffix = FAMILY_SUFFIX.get(e.family)
    if suffix is not None and not e.gold_segmentation.split("|")[-1].endswith(suffix):
        msgs.append(f"family {e.family} expects suffix {suffix!r}, got {e.gold_segmentation!r}")
    if "{w}" not in e.carrier:
        msgs.append("carrier missing {w} slot")
    if not e.base or not e.derived:
        msgs.append("empty base/derived")
    if e.lang not in {"ca", "en"}:
        msgs.append(f"bad lang {e.lang!r}")
    if e.morph_type not in {"derivational", "inflectional"}:
        msgs.append(f"bad morph_type {e.morph_type!r}")
    return msgs


def main() -> None:
    errors = []
    for e in LEXICON:
        errors += [f"{e.derived}: {m}" for m in validate_entry(e)]
    if errors:
        raise SystemExit("lexicon validation failed:\n" + "\n".join(errors))
    df = pd.DataFrame([vars(e) for e in LEXICON])
    out = ROOT / "data" / "morph_pairs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows, {df.family.nunique()} families)")


if __name__ == "__main__":
    main()
