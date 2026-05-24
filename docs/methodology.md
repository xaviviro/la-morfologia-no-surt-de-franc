# Metodologia

> Com mesurem la fragmentació de la morfologia catalana pels tokenitzadors i
> la seva conseqüència sobre la geometria de l'espai latent, i com provem si
> una segmentació conscient dels morfemes ("universal") recupera aquesta
> geometria sense reentrenar. El resum d'alt nivell és al
> [`../README.md`](../README.md); les ordres de reproducció són a
> [`reproduce.md`](reproduce.md).

Aquest és un estudi germà de [*Coca Is Not Cocaine*](https://github.com/xaviviro/coca-is-not-cocaine), que reutilitza el mateix
panell de models de pesos oberts però fa una pregunta de tokenització i
morfologia en lloc d'una de col·lisió cultural.

---

## 1. Preguntes de recerca

- **RQ1 — Descriptiva (nivell de tokenitzador, sense GPU).** Com segmenten la
  morfologia catalana els tokenitzadors anglo-dominants vs els català-aware?
  Quantifiquem la **fertilitat** (subparaules per paraula) en conjunts de
  paraules catalanes i angleses aparellats, i el **recall de frontera
  morfèmica** — quan una paraula té una frontera de morfema coneguda
  (`ràpida|ment`, `gat|et`, `cant|em`), el tokenitzador hi posa una frontera
  de token?

- **RQ2 — Conseqüència geomètrica (nivell de model, GPU).** La tokenització
  nadiua produeix un subespai morfològic **lineal i composicional**? Manllevem
  el marc del subespai lineal de Bolukbasi et al. (2016): els vectors de
  desplaçament per família `o_i = v(derivat_i) − v(base_i)` (la "direcció
  `-ment`" i les seves anàlogues) haurien de ser una *direcció consistent* si
  l'operació és composicional.

- **RQ3 — Mitigació / contrafactual (nivell de model, GPU).** Si imposem una
  segmentació conscient dels morfemes ("universal") en temps d'inferència,
  mantenint els pesos del model fixos, el subespai es torna més lineal / puja
  la precisió d'analogia? És a dir, *ajudaria una tokenització diferent, fins
  i tot sense reentrenar?*

---

## 2. El lèxic

Un lèxic curat a mà (`data/morph_pairs.csv`, 356 parells (base, derivat),
llicència CC-BY) amb **fronteres de morfema oracle**, revisat per l'autor pel
que fa a la correcció catalana (accents, la regla del femení per a `-ment`,
formes irregulars). Cada fila: `lang, family, base, derived,
gold_segmentation` (morfemes separats per barra), `morph_type, carrier`.

| llengua | família | n | morfologia |
| --- | --- | -: | --- |
| ca | `ment` | 40 | adjectiu → adverbi, sobre el **femení** (`ràpid → ràpida → ràpidament`) |
| ca | `dim_et` | 30 | diminutiu `-et/-eta` |
| ca | `agent_dor` | 25 | agentiu `-dor` |
| ca | `nom_cio` | 24 | nominalitzador `-ció` |
| ca | `plural` | 30 | plural `-s/-os` |
| ca | `verb_em` | 25 | 1a pl. present `-em` |
| ca | `gender_a` | 25 | gènere `-a` |
| ca | `gem_lla` | 12 | plural amb **ela geminada** `l·l` (`col·legi → col·legis`) |
| ca | `cedilla` | 13 | plural amb **ç** (`plaça → places`, `feliç → feliços`) |
| ca | `ny` | 12 | plural amb el **dígraf ny** (`muntanya → muntanyes`) |
| en | `ly` | 40 | adverbi `-ly` (paral·lel a `-ment`) |
| en | `agent_er` | 25 | agentiu `-er` |
| en | `nom_tion` | 25 | nominalitzador `-tion`/`-ation` |
| en | `plural_s` | 30 | plural `-s` |

`-ment` és el protagonista. Les famílies angleses són un *baseline* paral·lel:
permeten preguntar si un tokenitzador anglo-dominant representa la morfologia
anglesa de manera més composicional que la catalana.

Una única **frase portadora** metalingüística posa cada paraula en una
posició sintàctica uniforme independentment de la categoria gramatical —
català: *He llegit la paraula {w} en un llibre.*; anglès: *I read the word
{w} in a book.* Això elimina els confusors sintàctics lligats a la categoria
de l'embedding contextual.

---

## 3. Auditoria del tokenitzador (RQ1)

Per a cadascun dels **11 tokenitzadors del panell** (sense pesos de model,
sense GPU) i cada paraula derivada:

- **Fertilitat** `f(w)` = nombre de tokens subparaula per a `w` aïllada
  (`add_special_tokens=False`).
- **Recall de frontera** = proporció de fronteres de morfema oracle de la
  paraula que coincideixen amb una frontera de token, llegida del
  `offset_mapping` de caràcters del *fast tokenizer*. Una paraula sense
  frontera interna val 1.0 de manera vàcua.

**Robustesa dels offsets.** Un tokenitzador la implementació *fast* del qual
no exposa offsets alineats a caràcters (l'extrem final de l'últim token ≠ la
longitud en caràcters de la paraula — p. ex. offsets en espai de bytes) dona
una mètrica de frontera poc fiable; per a aquests casos emetem `NaN` en lloc
d'un 0 enganyós. Això afecta `deepseek-llm-67b-base`, els offsets del qual
són en bytes (una `à` de dos bytes desincronitza les posicions de caràcter);
la seva fertilitat continua sent vàlida i es reporta.

**Diferència de fertilitat CA−EN.** Per a cada tokenitzador reportem el *gap*
de fertilitat mitjana català−anglès amb un **IC 95 % per bootstrap de dues
mostres** (català i anglès són paraules diferents, no aparellades);
`out/fertility_gap_ci.csv`, via `m02_tokenize_audit.fertility_gap_ci`.

---

## 4. Extracció d'estats ocults (RQ2, RQ3) — el mecanisme contrafactual

No podem reentrenar un model amb un tokenitzador nou, així que provem la
pregunta més propera: **la representació existent del model es torna més
composicional quan l'entrada se segmenta per les fronteres de morfema?**
Imposem la segmentació **empalmant seqüències de token ids**, reutilitzant el
vocabulari i la taula d'embeddings propis del model.

Per a una paraula derivada `w` amb morfemes oracle `[m_1, …, m_k]` dins d'una
portadora `PREFIX {w} SUFIX`, muntem els `input_ids` així:

```
[bos?] + tok(PREFIX) + ids_de_la_paraula + tok(SUFIX)
```

- **Condició nadiua:** `ids_de_la_paraula = tok(" " + w)` — una sola peça.
- **Condició morfèmica:** `ids_de_la_paraula = tok(" " + m_1) ++ tok(m_2) ++ …`
  — la segmentació oracle.
- **Condició aleatòria (placebo):** el mateix nombre de peces que la morfèmica,
  però tallant per una posició interna **aleatòria que evita la frontera de
  morfema** (llavor determinista per paraula). És el control que distingeix
  "alinear amb els morfemes" de "trossejar diferent": si la morfèmica només
  supera la nadiua però no l'aleatòria, el guany no és específic dels morfemes.

La primera peça es tokenitza amb un espai al davant perquè tant la família
SentencePiece (`▁`) com la BPE de nivell de byte (`Ġ`) rebin el marcador
d'inici de paraula correcte; els morfemes següents es tokenitzen pelats
(continuació interna real, **sense artefacte d'espai**). Els ids del
`PREFIX`/`SUFIX` són **idèntics** entre les tres condicions, de manera que
l'única cosa que varia és la segmentació interna de la paraula — aïllant-ne
l'efecte.

La paraula base (`ràpid`) és monomorfèmica, sempre d'una sola peça, i és el
punt de referència compartit pels desplaçaments de les tres condicions.

Els estats ocults de cada capa demanada es promitgen (*mean-pool*) sobre la
regió de la paraula i es normalitzen L2. L'extracció fa servir
`dtype=bfloat16`, `device_map="auto"`, sense quantització — emmirallant el
`02_extract_embeddings.py` de l'estudi coca.

**El "tokenitzador universal" de referència és la segmentació de morfema oracle
(gold)** — el límit superior morfològicament perfecte.

### Models de geometria i capes

La geometria de l'espai vectorial corre sobre el tram petit (cap en una GPU
de 16 GB a bf16), reutilitzant els escombrats de capes per model de l'estudi
coca (la més profunda en negreta):

| Model | escombrat de capes |
| --- | --- |
| `google/gemma-2-2b` | 6, 15, **22** |
| `google/gemma-4-E2B` | 6, 15, **22** |
| `Qwen/Qwen2-1.5B` | 7, 17, **26** |
| `Qwen/Qwen3.5-4B-Base` | 9, 22, **32** |
| `BSC-LT/salamandra-2b` | 6, 14, **21** |

---

## 5. Mètriques de geometria

Per als vectors de desplaçament d'una família `o_i = v(derivat_i) − v(base_i)`:

- **Consistència de direcció** = cosinus mitjà de cada `o_i` respecte a la
  direcció mitjana de la família. 1.0 = tots els parells es mouen igual (una
  direcció de morfema neta).
- **Ràtio de variància PC1** = proporció de variància al primer component
  principal dels desplaçaments. 1.0 = els desplaçaments cauen en una sola
  línia.
- **Precisió d'analogia** (*leave-one-out*) = per a cada parell `j` exclòs,
  prediu `v(derivat_j) ≈ v(base_j) + mitjana(o_{i≠j})`; el `derivat`
  candidat més proper (cosinus) és `derivat_j`?

Les tres es computen per `(model, capa, família, condició)` per a **nadiu**,
**morfèmic** i **aleatori**, més **dos deltes** a la capa més profunda:

- `delta = morfèmic − nadiu` — ajuda la segmentació morfèmica respecte a la
  nadiua?
- `delta_vs_random = morfèmic − aleatori` — el guany és **específic dels
  morfemes** o l'obté qualsevol re-segmentació? (control placebo)

**Intervals de confiança i p-valors.** A cada delta de la capa més profunda
adjuntem un **IC 95 % per bootstrap (1000 rèpliques)** i un **p-valor
bilateral** (columnes `*_ci_lo`/`*_ci_hi`/`*_p` de `out/geometry_metrics.csv`):

- **Consistència de direcció:** es re-mostregen els parells amb reemplaçament i
  es recalcula l'estadístic de família amb els mateixos índexs (aparellat)
  (`geom_lib.bootstrap_delta_ci_p`).
- **Precisió d'analogia:** es bootstrapa la **correcció per parell**
  (`geom_lib.analogy_correct_per_pair` + `paired_bootstrap_ci_p`) en lloc de
  re-executar el *leave-one-out* sobre índexs re-mostrejats, perquè els índexs
  duplicats trencarien la identitat del veí més proper i esbiaixarien
  l'estadístic.

**Comparacions múltiples.** Com que hi ha moltes cel·les `(model, família)`,
els p-valors es corregeixen amb **Benjamini–Hochberg (FDR)** dins de cada
(delta, mètrica) sobre les cel·les de la capa més profunda (columnes `*_q`).
Els mapes de calor marquen amb `*` les cel·les amb **q < 0,05**.

**Deltes agregats.** A `out/geometry_aggregate_ci.csv` reportem el delta mitjà
per model (sobre les famílies catalanes) i per família (sobre els models) amb
un **IC per bootstrap** sobre les unitats agregades (`geom_lib.bootstrap_mean_ci`).

---

## 6. Figures

`out/figs/`: fertilitat CA vs EN (11 tokenitzadors), mapa de calor de recall
de frontera (tokenitzador × família), el mapa de la direcció `-ment` (PCA
nadiu vs morfèmic, per model de geometria), barres nadiu-vs-morfèmic per a
consistència de direcció i precisió d'analogia, i el scatter
tokenització→geometria. El text de tots els gràfics és en català. Vegeu
[`findings.md`](findings.md).

---

## Referències

Bibliografia completa i treball relacionat a [`references.md`](references.md).
Les més directes:

- Bolukbasi, Chang, Zou, Saligrama & Kalai (2016). *Man is to computer
  programmer as woman is to homemaker? Debiasing word embeddings.*
  [arXiv:1607.06520](https://arxiv.org/abs/1607.06520)
- Ada Lovelace Institute (2025). *Tokenising culture.*
  [adalovelaceinstitute.org](https://www.adalovelaceinstitute.org/blog/cultural-misalignment-llms/)
- Vinaixa Roselló (2026). [*Coca Is Not Cocaine*](https://github.com/xaviviro/coca-is-not-cocaine) (estudi germà).
