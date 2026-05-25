# Resultats

> Les xifres són la sortida determinista dels scripts executats sobre el lèxic
> d'aquest repositori (**517 parells (base, derivat)**, 27 famílies, ca/es/en). Les xifres de
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

| grup | taxa de recuperació (*recall*) de frontera de `-ment` |
| --- | --: |
| anglo-dominants | 0,218 |
| BSC català-aware | 0,200 |
| DeepSeek 67B | n/d (desplaçaments inservibles) |

Tots dos grups aïllen el sufix `-ment` només ~20% de les vegades. **El
tokenitzador conscient del català fragmenta menys el català, però no talla per les
fronteres de morfema més sovint que l'anglo-dominant** —sobretot manté
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
català —coherent amb el fet que el preentrenament i la tokenització
anglo-dominants representin la morfologia anglesa de manera més composicional
(`out/figs/direction_consistency_bars.png`).

## 4. Una segmentació de morfema oracle recupera la geometria composicional

El contrafactual controlat (les mateixes paraules, resegmentades per les
fronteres de morfema oracle, pesos fixos) millora la geometria de mitjana en els
**cinc** models. `delta = morfèmic − nadiu` mitjà sobre les 14 famílies
catalanes amb tall morfèmic (capa més profunda):

| model | Δ consistència de direcció | Δ precisió d'analogia |
| --- | --: | --: |
| `gemma-2-2b` | +0,108 | +0,047 |
| `gemma-4-E2B` | +0,037 | +0,074 |
| `Qwen2-1.5B` | +0,049 | +0,030 |
| `Qwen3.5-4B-Base` | +0,071 | +0,139 |
| `salamandra-2b` | +0,199 | +0,057 |

**Els deu deltes (5 models × 2 mètriques) són positius.** Sota la hipòtesi nul·la
que cada signe fos una moneda justa, això és un **test de signes** amb
**p = 0,0020** (exacte, bilateral; `geom_lib.sign_test`) — una de les evidències
més fortes de l'estudi, ara reportada formalment. **Salamandra-2B —el model
conscient del català— és el que més guanya en consistència de direcció**
(+0,199): el seu tokenitzador fragmenta menys el català, però imposar la
segmentació *alineada amb els morfemes* encara recupera una direcció força més
neta, reforçant §2 (menys fragmentació no és alineació morfèmica). Vegeu
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
però amb només ~40 parells els **IC del 95% mitjançant bootstrap aparellat** mostren que
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

Representar la consistència de direcció nadiua contra la taxa de recuperació de frontera
nadiua per `(model, família)` **no** dona una correlació positiva neta
(`out/figs/tok_vs_geom_scatter.png`). El motiu és un confusor de la mètrica:
`taxa de recuperació de frontera = 0` barreja una paraula que rep el **seu propi token
únic** (neta) i una paraula partida **lluny** de la seva frontera de morfema
(bruta). El contrafactual controlat intra-paraula de §4 és l'evidència causal
més neta; el gràfic de dispersió (*scatter*) es reporta per transparència, no com a suport.

## 6. Trets ortogràfics catalans: el punt volat castiga el doble

Tres famílies de plural addicionals aïllen un tret ortogràfic característic del
català al radical, amb la morfologia (plural) constant:

| família | exemple | fertilitat anglo-dominant | fertilitat BSC |
| --- | --- | --: | --: |
| `gem_lla` — ela geminada `l·l` | col·legi → col·legis | **4,58** | **4,00** |
| `cedilla` — c trencada `ç` | plaça → places | 2,22 | 1,77 |
| `ny` — dígraf `ny` | muntanya → muntanyes | 2,40 | 1,83 |
| `plural` (referència) | gat → gats | 2,05 | 1,33 |

El **punt volat és el cas extrem**: tots dos tokenitzadors aïllen el punt volat (`·`) com a
token propi (`col · leg is`), de manera que una paraula amb `l·l` costa ~4–4,6
subparaules —gairebé el doble d'un plural normal, i **ni el tokenitzador
conscient del català ho comprimeix** (4,00 vs 4,58). La `ç` i el dígraf `ny` afegeixen
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

## 8. Un segmentador realista (Morfessor) ja recupera gairebé tot el guany

L'oracle és el sostre teòric. Per saber què en recuperaria una eina
**desplegable**, afegim un **Morfessor** no supervisat (entrenat només sobre les
cadenes del lèxic, mai veu les fronteres gold). Recupera el **53,8 %** de les
fronteres morfèmiques (precisió 52,5 %; `out/morfessor_agreement.csv`,
`out/figs/morfessor_agreement.png`) — un segmentador a mig camí entre el nadiu
(0 %) i l'oracle (100 %).

Tot i així, geomètricament gairebé **iguala l'oracle**. `Δ consistència de
direcció` (Morfessor − nadiu) vs (oracle − nadiu), mitjana famílies CA, IC 95 %:

| model | oracle − nadiu | Morfessor − nadiu |
| --- | --- | --- |
| `gemma-2-2b` | +0,108 [+0,059, +0,156] | +0,097 [+0,052, +0,142] |
| `gemma-4-E2B` | +0,037 [−0,004, +0,077] | **+0,052 [+0,024, +0,081]** |
| `Qwen2-1.5B` | +0,049 [−0,005, +0,102] | **+0,053 [+0,005, +0,099]** |
| `Qwen3.5-4B-Base` | +0,071 [+0,032, +0,114] | +0,060 [+0,016, +0,105] |
| `salamandra-2b` | +0,199 [+0,148, +0,249] | +0,140 [+0,093, +0,190] |

El **delta de Morfessor exclou el zero en els cinc models** — fins i tot a
`gemma-4-E2B` i `Qwen2-1.5B`, on l'oracle no arribava a la significació. La
lectura pràctica és forta: **no cal una segmentació morfològicament perfecta**;
un segmentador no supervisat estàndard ja exposa la major part de l'estructura
composicional. L'escala de segmentació
(`out/figs/condition_ladder.png`) ho resumeix: nadiu (0,55) → **aleatori cau a
0,50** → Morfessor no supervisat (0,63) ≈ **regles català** (0,63) ≈ oracle
(0,64). El **segmentador de regles català** (`scripts/rule_seg.py`) —el
condicionant "desplegable" que un analitzador de regles representaria— encerta
el **77,9 %** de les fronteres gold (vs 53,8 % de Morfessor) i iguala l'oracle
en geometria: és la condició realista més forta de l'estudi.

## 9. Robustesa entre capes

El guany no depèn de la tria de capa. `Δ (oracle − nadiu)` de consistència de
direcció (mitjana famílies CA) és **positiu a les tres profunditats escombrades
en els cinc models** (p. ex. `gemma-2-2b` L6 +0,115 → L15 +0,087 → L22 +0,108;
`salamandra-2b` L6 +0,226 → L14 +0,187 → L21 +0,199), sense canviar de signe
(`out/figs/layer_robustness.png`). Els
IC i p-valors es calculen ara a totes les capes, no només a la més profunda.

## 10. Regularitat morfèmica indoeuropea: es manté, però incompleta i uniforme

La teoria morfològica (productivitat de Baayen; flexió > derivació) prediria que
els morfemes **flexius** són més regulars que els **derivatius** i, per tant,
geometria més neta. Provem aquesta hipòtesi amb les 14 famílies catalanes amb
tall morfèmic (incloses prefixos i `-ció` profund), etiquetades a priori a
`data/family_traits.csv` (vegeu
[`morphology-background.md`](morphology-background.md) §5).

Consistència de direcció (mitjana sobre models, capa profunda), flexió vs
derivació, per condició de segmentació:

| condició | global | flexió | derivació | diff (flexió − deriv.) [IC 95 %] |
| --- | --: | --: | --: | --- |
| nadiu | 0,55 | 0,497 | 0,591 | −0,094 [−0,208, +0,019] |
| morfèmic (oracle) | 0,64 | 0,599 | 0,676 | −0,077 [−0,159, +0,014] |
| Morfessor | 0,63 | 0,568 | 0,678 | −0,110 [−0,201, −0,016] |

**La hipòtesi no es confirma — i de fet apunta al contrari.** La diferència
flexió−derivació és nul·la o lleugerament *negativa* (la derivació és igual o
**més** consistent), i amb Morfessor la diferència ja exclou el zero a favor de
la derivació; el Spearman amb el rang de transparència és **positiu** (+0,4 a
+0,5), el contrari del que predeia la productivitat. Tres lectures honestes:

1. **La regularitat morfèmica es manté a escala, però és incompleta.** Amb el
   tall morfèmic la consistència convergeix a **~0,64** (no a 1,0). La "desviació
   per arbitrarietat" (canvi fonètic regular, gramaticalització, lexicalització;
   §3 del rerefons) és **real i substancial** però residual i clarament
   mesurable.

2. **La derivació no és menys regular que la flexió; si de cas, ho és més.** En
   aquests models l'esquema clàssic "flexió més productiva ⇒ més regular" **no
   es trasllada** a la geometria. La derivació catalana (prefixos i sufixos
   transparents, `-ització` ultraproductiu) viu en direccions tan netes o més
   que la flexió.

3. **El que separa la flexió en nadiu és tokenització, no llengua.** La flexió
   queda enrere en nadiu (0,50 vs 0,59) però no amb el tall morfèmic: els afixos
   flexius curts es fragmenten de manera que perjudica la geometria nadiua, i
   per això es recuperen més (§4). La geometria nadiua barreja morfologia amb
   tokenització; només el tall morfèmic aïlla la regularitat del morfema.

> **Confusió de la composició del grup.** El grup "flexió" està dominat per
> **plurals**: `plural`, `gem_lla`, `cedilla` i `ny` són tots plurals, i amb
> `gender_a` i `verb_em` la categoria és de facto "plural+gènere+1pl vs tota la
> resta". El contrast flexió-vs-derivació és, doncs, parcialment un contrast
> *plural vs no-plural*; el resultat "la hipòtesi no es confirma" és real, però
> aquesta confusió de composició el matisa i és per això que ens recolzem en el
> contrast per condició (nadiu/morfèmic), no en l'etiqueta flexiu/derivatiu sola.

Vegeu `out/figs/regularity.png`, `out/regularity_analysis.csv` i
`out/regularity_summary.csv`.

## 11. Patrons indoeuropeus: prefixació, gradient de Sturtevant, profunditat

Tres patrons motivats per la morfologia IE
([`morphology-background.md`](morphology-background.md)), tots a la capa més
profunda amb IC 95 % per bootstrap:

**A · Prefixació (el buit estructural).** Tot l'estudi era sufixal; afegim
prefixos (`des-`, `re-`, `in-/im-`). El seu Δ morfèmic de consistència de
direcció és **+0,198**, enorme i molt superior al dels sufixos derivatius
(+0,036): diff **+0,162 [+0,107, +0,212], p = 0,001**. És el delta més gran de
tot l'estudi — els prefixos catalans són els que la tokenització nadiua
fragmenta de manera més perjudicial, i els que més recupera el tall morfèmic.

**B/C · Gradient de Sturtevant (regular vs irregular).** La predicció
neogramàtica era que la consistència de direcció **decaigués** regular →
alternança d'arrel → suppletiu (verbs, present 1a sg). El resultat és **feble i
no monòton**: verb regular 0,554 [0,495, 0,609] > suppletiu 0,517 [0,454, 0,586]
> alternança 0,498 [0,461, 0,532]. Hi ha una *tendència* que els regulars siguin
més consistents que els irregulars, coherent amb Sturtevant, però els IC se
solapen i el suppletiu no segueix el gradient — amb n petita (8–15 verbs) **no
es pot afirmar un gradient net** (`out/figs/sturtevant_gradient.png`).

**D · Profunditat derivativa (FALSADOR).** La predicció era que `-ció` perdés
consistència quan la base ja és derivada. **Surt el contrari**: `-ció` sobre
base derivada (`globalitzar→globalització`, d1 = 0,831) és **més** consistent
que sobre base simple (`crear→creació`, d0 = 0,694): diff **−0,137 [−0,215,
−0,064], p = 0,001**. La profunditat no degrada la composicionalitat; el que
mana és la **regularitat del patró**, i `-ització` és extraordinàriament
sistemàtic i productiu. Una capa morfològica de més no és soroll si el patró és
regular.

Artefactes: `out/ie_sturtevant_gradient.csv`, `out/ie_patterns_summary.csv`.

## 12. Robustesa de portadora: el guany no és un artefacte de la menció

Tots els resultats anteriors fan servir una portadora de **menció** (*He llegit
la paraula {w}…*). Per descartar que la geometria sigui un artefacte d'aquest
marc, re-extraiem tot sota una segona portadora d'**ús** per categoria
(p. ex. *Ho fa {w} cada dia* per a adverbis; *Veig {w} pertot arreu* per a
plurals) i recalculem el delta morfèmic (oracle − nadiu) per `(model, família)`.

| comparació menció vs ús | valor |
| --- | --- |
| Δ mitjà (menció) | +0,093 |
| Δ mitjà (ús) | +0,078 |
| concordança de signe | **61/70 (87 %)** |
| correlació de Spearman | **+0,928** |
| Δ d'ús positiu | 56/70 (test de signes **p < 0,001**) |

El delta sota la portadora d'ús **segueix gairebé idènticament** el de menció
(Spearman +0,93; els punts cauen sobre la diagonal a
`out/figs/carrier_robustness.png`). La millora morfèmica **no depèn de la
portadora**. (Les famílies `dim_et` i `verb_supl` usen una portadora d'ús
imperfecta per gènere/heterogeneïtat; vegeu [`limitations.md`](limitations.md).)

## 13. Rèplica romànica: el patró català es manté en castellà

Sis famílies castellanes reflecteixen sis de catalanes (`-mente↔-ment`,
`-ción↔-ció`, `-dor↔-dor`, dim. `-ito↔-et`, plural, gènere). Comparant-les a la
capa més profunda:

- Consistència de direcció nadiua: **Spearman(CA, ES) = +0,886**.
- Delta morfèmic (oracle − nadiu): **Spearman(CA, ES) = +0,771**.

Les famílies que tenen geometria neta (o que guanyen amb el tall morfèmic) en
català la tenen també en castellà, i a la inversa (`out/romance_comparison.csv`).
El fenomen **no és una idiosincràsia del català**: es replica en una segona
llengua romànica. Això sosté el marc **indoeuropeu** —dues romàniques (català,
castellà) més una germànica (anglès, *baseline*)— però **no** una universalitat
plena: l'IE és tipològicament molt més divers (eslau, indi, grec…) i tres
llengües en són una mostra petita (vegeu [`limitations.md`](limitations.md)).

---

## Titular

El català es tokenitza ~1,7× més fi que l'anglès als models anglo-dominants
(i fins a 1,9× a DeepSeek-67B); un vocabulari conscient del català (BSC
Salamandra/ALIA) ho redueix a 1,36×. Però menys fertilitat no implica
alineació morfèmica —el sufix `-ment` només s'aïlla ~20% de les vegades en
*qualsevol* tokenitzador— i la morfologia anglesa parteix de manera més composicional
que la catalana a l'estat ocult. Imposar una segmentació de morfema oracle en
temps d'inferència recupera la geometria composicional en els cinc models petits,
cosa que suggereix que una tokenització conscient dels morfemes exposaria una
estructura composicional que aquests models ja codifiquen en part. El guany és
**específic dels morfemes** (cau amb talls aleatoris), **no cal un oracle** (un
segmentador de regles català, recall 0,78, o fins i tot Morfessor no supervisat
ja l'igualen), **no depèn de la portadora** (es replica sota un marc d'ús,
Spearman +0,93) i **es manté en castellà** (Spearman CA~ES +0,89) — un patró
estable a través de tres llengües indoeuropees, sense pretensió d'universalitat
plena. Vegeu [`limitations.md`](limitations.md) per a què afirma i què no afirma
això, incloent-hi que el contrafactual mesura recombinació de peces BPE
alineades amb morfemes, no codificació del morfema com a unitat.
