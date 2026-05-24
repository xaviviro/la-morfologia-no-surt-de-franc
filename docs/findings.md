# Resultats

> Les xifres són la sortida determinista dels scripts comesos sobre el lèxic
> comès (**319 parells (base, derivat)**, 11 famílies). Les xifres de
> geometria són a la capa més profunda escombrada de cada model si no
> s'indica el contrari. Vegeu [`methodology.md`](methodology.md) per a les
> definicions de les mètriques.

---

## 1. El català paga un peatge de fragmentació; els tokenitzadors català-aware el redueixen

Fertilitat mitjana (subparaules per paraula) sobre el lèxic curat, i la ràtio
català-sobre-anglès:

| grup de tokenitzador | català | anglès | ràtio CA/EN |
| --- | --: | --: | --: |
| anglo-dominants (Gemma, Qwen, Mistral) | 2,39 | 1,43 | **1,67×** |
| BSC català-aware (Salamandra 2B/7B, ALIA 40B) | 1,64 | 1,21 | **1,36×** |
| DeepSeek 67B | 2,36 | 1,25 | 1,89× |

Tots els tokenitzadors fragmenten més el català que l'anglès
(`out/figs/fertility_ca_vs_en.png`, `out/figs/fertility_ratio.png`). El
càstig més gran per al català és DeepSeek-67B (1,89×) i Gemma (~1,85×); el
més baix de bon tros és el tram català-aware del BSC (1,36×) — el seu
vocabulari, entrenat amb una proporció no trivial de català, gasta ~1,64
subparaules en una paraula catalana davant de ~2,4 del tram anglo-dominant.
És el resultat descriptiu més net i exactament el que hauria de comprar un
vocabulari conscient del català.

Per família, la derivació amb sufixos llargs es fragmenta més al tram
anglo-dominant (`out/figs/fertility_by_family.png`).

## 2. Menys fragmentació ≠ alineació morfèmica

Recall mitjà de frontera de `-ment` (cau un tall de token a la frontera
`tall|ment`?):

| grup | recall de frontera de `-ment` |
| --- | --: |
| anglo-dominants | 0,218 |
| BSC català-aware | 0,200 |
| DeepSeek 67B | n/d (offsets inservibles) |

Tots dos grups aïllen el sufix `-ment` només ~20 % de les vegades. **El
tokenitzador català-aware fragmenta menys el català, però no talla per les
fronteres de morfema més sovint que l'anglo-dominant** — sobretot manté
paraules senceres en un sol token, amagant la costura del morfema a dins.
La il·lustració de `out/figs/tokenization_examples.png` ho fa concret: Gemma
parteix *ràpidament* com `rà|pid|ament` (la costura real `ràpida|ment` cau
*dins* de l'últim token), mentre que Salamandra manté *ràpidament* com un sol
token — cap dels dos exposa el morfema `-ment`. Tots dos sí que alineen a
*gatet* (`gat|et`) i Salamandra alinea a *cantem* (`cant|em`).

## 3. La morfologia anglesa parteix més composicional que la catalana

Consistència de direcció i precisió d'analogia nadiues, `-ment` vs la seva
anàloga anglesa `-ly` (capa més profunda, entre models):

- `-ly` consistència de direcció nadiua ≈ 0,62–0,78, analogia ≈ 0,90–0,97.
- `-ment` consistència de direcció nadiua ≈ 0,48–0,63, analogia ≈ 0,63–0,80.

El sufix adverbial anglès ja viu en un subespai nadiu més net i lineal que el
català — coherent amb el fet que el preentrenament i la tokenització
anglo-dominants representin la morfologia anglesa de manera més composicional
(`out/figs/direction_consistency_bars.png`).

## 4. Una segmentació de morfema oracle recupera geometria composicional

El contrafactual controlat (les mateixes paraules, re-segmentades per les
fronteres de morfema oracle, pesos fixos) millora la geometria de mitjana als
**cinc** models. `delta = morfèmic − nadiu` mitjà sobre les 7 famílies
catalanes (capa més profunda):

| model | Δ consistència de direcció | Δ precisió d'analogia |
| --- | --: | --: |
| `gemma-2-2b` | +0,104 | +0,035 |
| `gemma-4-E2B` | +0,028 | +0,065 |
| `Qwen2-1.5B` | +0,026 | +0,036 |
| `Qwen3.5-4B-Base` | +0,063 | +0,151 |
| `salamandra-2b` | +0,180 | +0,006 |

Els deu deltes són positius. **Salamandra-2B — el model català-aware — és el
que més guanya en consistència de direcció** (+0,180): el seu tokenitzador
fragmenta menys el català, però imposar la segmentació *alineada amb els
morfemes* encara recupera una direcció força més neta, reforçant §2 (menys
fragmentació no és alineació morfèmica). Vegeu
`out/figs/delta_heatmap_direction.png` i
`out/figs/delta_heatmap_analogy.png`.

### Per família (Δ mitjà entre models, capa més profunda)

| família | Δ consistència de direcció | Δ precisió d'analogia |
| --- | --: | --: |
| `plural` | +0,182 | +0,187 |
| `gender_a` | +0,159 | +0,040 |
| `dim_et` | +0,097 | +0,007 |
| `verb_em` | +0,077 | +0,088 |
| `ment` | +0,039 | +0,075 |
| `agent_dor` | +0,026 | +0,016 |
| `nom_cio` | −0,019 | +0,000 |

L'efecte és **més fort en la morfologia flexiva** (plural, gènere) i positiu
per a `-ment`. **No és perfectament uniforme**: `nom_cio` (el nominalitzador
`-ció`) mostra un petit delta de direcció negatiu. La lectura honesta és "la
segmentació morfèmica ajuda de mitjana, sobretot en flexió", no "sempre ajuda
arreu".

### `-ment` en concret, amb intervals de confiança

Re-segmentar els adverbis en `-ment` mou els punts en la direcció esperada
(consistència de direcció amunt en 4 de 5 models, analogia amunt en 4 de 5),
però amb només ~40 parells els **IC 95 % per bootstrap aparellat** mostren que
no totes les cel·les són individualment distingibles de zero:

| model | Δ consistència de direcció [IC 95 %] | Δ precisió d'analogia [IC 95 %] |
| --- | --- | --- |
| `gemma-2-2b` | +0,056 [+0,007, +0,097] * | +0,100 [−0,025, +0,225] |
| `gemma-4-E2B` | +0,012 [−0,025, +0,050] | −0,225 [−0,400, −0,025] * |
| `Qwen2-1.5B` | +0,036 [−0,014, +0,082] | +0,125 [+0,000, +0,250] |
| `Qwen3.5-4B-Base` | +0,096 [+0,027, +0,153] * | +0,050 [−0,100, +0,200] |
| `salamandra-2b` | −0,004 [−0,041, +0,042] | +0,325 [+0,175, +0,475] * |

`*` = l'IC exclou el zero. El guany d'analogia més fort i clarament
significatiu és el de Salamandra-2B (**+0,325**); l'única **regressió**
significativa és l'analogia de Gemma-4-E2B (**−0,225**). La consistència de
direcció puja de manera significativa en `gemma-2-2b` i `Qwen3.5-4B`. La resta
de cel·les de `-ment` són positives de mitjana però els seus IC creuen el zero
— senyal real però modest a aquesta mida de mostra. Vegeu
`out/figs/ment_delta_forest.png` (forest plot) i `out/figs/ment_summary.png`;
els mapes de direcció (`out/figs/ment_direction_*.png`) mostren les fletxes
morfèmiques visiblement més paral·leles que les nadiues.

### Significació a tot el panell

Els mapes de calor del delta (`out/figs/delta_heatmap_*.png`) marquen amb un
asterisc les cel·les `(model, família)` l'IC 95 % de les quals exclou el zero
(IC per parell per a l'analogia; bootstrap sobre parells re-mostrejats per a la
consistència de direcció, totes dues a la capa més profunda). Els efectes més
robustos i repetits són en **flexió** (plural, gènere, `verb_em`); els deltes
negatius significatius són escassos i dispersos. Els valors d'IC per cel·la
són a `out/geometry_metrics.csv` (columnes `*_ci_lo` / `*_ci_hi`).

## 5. El vincle transversal tokenització→geometria està confós

Representar la consistència de direcció nadiua contra el recall de frontera
nadiu per `(model, família)` **no** dona una correlació positiva neta
(`out/figs/tok_vs_geom_scatter.png`). El motiu és un confusor de la mètrica:
`recall de frontera = 0` barreja una paraula que rep el **seu propi token
únic** (neta) i una paraula partida **lluny** de la seva frontera de morfema
(bruta). El contrafactual controlat intra-paraula de §4 és l'evidència causal
més neta; el scatter es reporta per transparència, no com a suport.

## 6. Trets ortogràfics catalans: el punt volat castiga el doble

Tres famílies de plural addicionals aïllen un tret ortogràfic característic del
català al radical, amb la morfologia (plural) constant:

| família | exemple | fertilitat anglo | fertilitat BSC |
| --- | --- | --: | --: |
| `gem_lla` — ela geminada `l·l` | col·legi → col·legis | **4,58** | **4,00** |
| `cedilla` — c trencada `ç` | plaça → places | 2,22 | 1,77 |
| `ny` — dígraf `ny` | muntanya → muntanyes | 2,40 | 1,83 |
| `plural` (referència) | gat → gats | 2,05 | 1,33 |

El **punt volat és el cas extrem**: tots dos tokenitzadors aïllen el `·` com a
token propi (`col · leg is`), de manera que una paraula amb `l·l` costa ~4–4,6
subparaules — gairebé el doble d'un plural normal, i **ni el tokenitzador
català-aware ho comprimeix** (4,00 vs 4,58). La `ç` i el dígraf `ny` afegeixen
un càstig menor; Gemma arriba a aïllar la `ç` (`ca ç ador`) i fins i tot a
partir el dígraf (`an ys`), mentre que Salamandra els manté més sovint
(`caç ador`, `anys`). Vegeu `out/figs/tokenization_examples_ortho.png`.

**Geometria.** Imposar la segmentació morfèmica també ajuda aquestes famílies,
sobretot en consistència de direcció: `gem_lla` (l·l) **+0,093**, `ny` **+0,082**,
`cedilla` (ç) +0,021 de mitjana entre models — coherent amb la idea que les
paraules pitjor fragmentades són les que més guanyen quan s'alinea el tall amb
els morfemes. Els efectes d'analogia són més sorollosos i poques cel·les
individuals tenen l'IC 95 % que exclou el zero (vegeu els asteriscos als mapes
de calor).

## 7. Rigor: el guany és específic dels morfemes (control placebo)

El dubte evident sobre §4 és si *qualsevol* re-segmentació canvia la geometria,
no l'alineació amb els morfemes. El **control placebo** ho descarta: comparem la
condició morfèmica amb una condició **aleatòria** (mateix nombre de peces, però
tallant per una posició interna no morfèmica).

`Δ consistència de direcció` (mitjana sobre famílies CA, capa més profunda, IC 95 %):

| model | morfèmic − nadiu | morfèmic − aleatori |
| --- | --- | --- |
| `gemma-2-2b` | +0,088 [+0,045, +0,136] | +0,149 [+0,109, +0,194] |
| `gemma-4-E2B` | +0,029 [−0,011, +0,070] | +0,077 [+0,026, +0,139] |
| `Qwen2-1.5B` | +0,026 [−0,032, +0,080] | +0,129 [+0,081, +0,176] |
| `Qwen3.5-4B-Base` | +0,054 [+0,024, +0,089] | +0,150 [+0,100, +0,208] |
| `salamandra-2b` | +0,181 [+0,127, +0,218] | +0,208 [+0,172, +0,245] |

En els **cinc models** la segmentació morfèmica supera l'aleatòria amb un IC que
exclou el zero, i sovint per més marge que respecte a la nadiua. És a dir, la
segmentació **aleatòria empitjora la geometria per sota de la nativa**, mentre
que l'alineada amb els morfemes la millora: **el guany és específic dels
morfemes**, no de trossejar diferent. Vegeu `out/figs/placebo_control.png`.

**Comparacions múltiples (Benjamini–Hochberg, cel·les de la capa més profunda):**

| contrast | direcció (q<0,05) | analogia (q<0,05) |
| --- | --: | --: |
| morfèmic − nadiu | 28/70 | 7/70 |
| morfèmic − aleatori | 49/70 | 26/70 |

El contrast placebo és el més robust fins i tot després de corregir per
comparacions múltiples.

**Fertilitat CA−EN amb IC.** El *gap* de fertilitat català−anglès té un IC 95 %
que exclou el zero en **els 11 tokenitzadors** (DeepSeek-67B +1,21 [+1,08, +1,33]
a dalt; els BSC ~+0,57 [+0,46, +0,69] a baix), confirmant §1 amb incertesa
(`out/fertility_gap_ci.csv`).

---

## Titular

El català es tokenitza ~1,7× més fi que l'anglès als models anglo-dominants
(i fins a 1,9× a DeepSeek-67B); un vocabulari català-aware (BSC
Salamandra/ALIA) ho redueix a 1,36×. Però menys fertilitat no implica
alineació morfèmica — el sufix `-ment` només s'aïlla ~20 % de les vegades en
*qualsevol* tokenitzador — i la morfologia anglesa parteix més composicional
que la catalana a l'estat ocult. Imposar una segmentació de morfema oracle en
temps d'inferència recupera geometria composicional als cinc models petits,
sobretot en flexió, cosa que suggereix que una tokenització conscient dels
morfemes ("universal") exposaria estructura composicional que aquests models
ja codifiquen en part. Vegeu [`limitations.md`](limitations.md) per a què
afirma i què no afirma això.
