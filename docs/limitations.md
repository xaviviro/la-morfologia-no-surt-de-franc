# Limitacions

Què **no** afirma aquest estudi, i les advertències que el lector ha de tenir
presents.

## El contrafactual és una sonda en inferència, no un estudi de reentrenament

La condició morfèmica imposa una segmentació de morfema oracle empalmant identificadors de token (*token IDs*), reutilitzant el vocabulari i la taula d'embeddings existents del model
(vegeu [`methodology.md`](methodology.md) §4). El model **no es va entrenar**
amb aquesta segmentació. Per tant:

- Un resultat **positiu** (geometria morfèmica més neta que la nadiua) mostra
  que les peces composicionals ja són presents a la representació, i que un
  tokenitzador conscient dels morfemes les exposaria. **No** demostra que un
  model *reentrenat* amb aquest tokenitzador arribi a aquesta geometria —
  només que els pesos actuals ho permeten.
- Un resultat **nul/negatiu** per a una família donada **no** descarta que un
  model *reentrenat* amb un tokenitzador conscient dels morfemes en
  surti beneficiat. Això no ho podem observar aquí.

És deliberadament un diagnòstic intrínsec i sense entrenament. Reentrenar un
tokenitzador + model queda fora d'abast. El **control placebo** (condició
aleatòria, findings §7) sí que descarta que el guany sigui de "trossejar
diferent" en lloc d'alinear amb els morfemes, i la significació per cel·la està
corregida per comparacions múltiples (FDR); però cap d'aquestes coses no
substitueix un estudi de reentrenament.

## La peça BPE no és el morfema

El caveat més profund (vegeu [`methodology.md`](methodology.md) §4, "Què mesura
i què no"): l'embedding de la peça `ment` és el seu vector distribucional après
de *element, moment, argument*…, **no "el morfema -ment"**. El contrafactual
mesura *recombinació composicional de peces BPE alineades amb morfemes*, no
*codificació del morfema com a unitat*. El títol s'ha de llegir en aquest sentit
restringit.

## Asimetria de pooling base/derivat

La base és sempre d'una peça i el derivat morfèmic de *k* peces *mean-pooled*.
En el delta `morfèmic − nadiu` la base es cancel·la, però el nombre de peces
poolejades pot diferir entre condicions, i les mètriques **absolutes**
(regularitat §10, gradient §11) comparen un vector pooled-de-k amb un
pooled-d'1. Un control net pendent seria igualar el nombre de peces poolejades
entre condicions (o controlar el delta per Δ del nombre de peces); ara mateix
"nadiu" i "morfèmic" difereixen en dues coses alhora (fronteres **i** nombre de
peces).

## Una sola frase portadora, i metalingüística

Totes les extraccions fan servir una única portadora de **menció** (*He llegit
la paraula {w} en un llibre*), no d'**ús**. Les representacions de paraules
citades poden ser atípiques. Faltaria una comprovació de robustesa amb 2–3
portadores (alguna d'ús normal) per descartar que la geometria sigui un
artefacte de la portadora.

## Estudi intrínsec, sense ancoratge a tasca

Tot l'estudi és **geometria**, no comportament. La pregunta "i això quant canvia
el rendiment?" queda oberta. El pont seria una tasca de *probing* morfològic
(concordança de gènere/nombre, *cloze* morfològic) sota les tres condicions, i
la connexió fertilitat→rendiment de Goldman et al. (2024). És l'addició
d'impacte més gran pendent.

## El "tokenitzador universal" és un oracle, no un sistema desplegable

La segmentació de morfema és la frontera oracle curada per construcció al
lèxic — un límit superior morfològicament perfecte. Un segmentador universal
real (p. ex. Morfessor, un analitzador morfològic del català) seria més
sorollós. El nostre resultat és doncs "la segmentació alineada amb els
morfemes ajuda al sostre", cosa que acota, però no iguala, el que donaria una
eina pràctica.

## La taxa de recuperació de frontera = 0 és ambigua

Una paraula obté un 0 en la taxa de recuperació de frontera tant quan és un **token únic**
(sense tall intern —sovint una representació *neta*) com quan es parteix
**lluny** de la seva frontera de morfema (una de *bruta*). La mètrica
descriptiva de recuperació (*recall*) no separa aquests casos, i per això el gràfic de dispersió (*scatter*)
transversal tokenització→geometria està confós
([`findings.md`](findings.md) §5). El contrafactual controlat intra-paraula
no es veu afectat per això i és l'evidència causal primària.

## La taxa de recuperació de frontera de DeepSeek no està disponible

El *fast tokenizer* de `deepseek-llm-67b-base` retorna desplaçaments (*offsets*) en espai de
bytes (no de caràcters), de manera que la seva taxa de recuperació de frontera morfèmica no
es pot computar de manera fiable i es reporta com a `NaN`. La seva fertilitat
és vàlida.

## Abast

- **Només models base / pre-RLHF** (com a l'estudi coca). Les variants
  instruïdes/xat poden diferir.
- **Geometria només al tram petit** (5 models ≤ 4B). L'auditoria cobreix els
  11 tokenitzadors, però la geometria de l'espai vectorial no s'ha corregut
  al tram gran.
- **L'elecció de capa** reutilitza els escombrats per model de l'estudi coca;
  s'usa la capa més profunda escombrada per a les xifres titulars. Els
  resultats a les tres capes escombrades són a `out/geometry_metrics.csv`.
- **La mida del lèxic** és de 441 parells (8–40 per família); les estimacions
  per família són més estables que la primera passada de ~150 parells però
  encara modestes. Per això els deltes per cel·la `(model, família)` ara porten
  un **IC 95 % per bootstrap** (a la capa més profunda; vegeu
  [`methodology.md`](methodology.md) §5), i molts IC individuals creuen el zero
  malgrat un patró agregat positiu — és a dir, el senyal és real de mitjana
  però sovint no és distingible de zero cel·la a cel·la a aquesta `n`.
- **Peculiaritats del baseline anglès.** Algunes cel·les angleses es comporten
  de manera estranya (p. ex. l'analogia nadiua de `-ly` a `gemma-4-E2B` és
  insòlitament baixa, 0,25, i salta a 0,85 amb la re-segmentació); amb una `n`
  per família petita i artefactes específics del tokenitzador, no s'han de
  sobreinterpretar cel·les individuals.

## Anàlisi de regularitat (flexió vs derivació)

L'anàlisi de [`findings.md`](findings.md) §10 té dues limitacions. Primer, hi ha
només **14 famílies catalanes amb tall morfèmic**, així que els tests
flexió-vs-derivació tenen poca potència (els IC són amples — un *nul* amb n
petita no és una prova forta d'absència d'efecte). Segon, les etiquetes de
`data/family_traits.csv` (morph_type, transparency_rank) es basen en principis
lingüístics estàndard, però **l'anotador ja coneixia els resultats geomètrics**,
de manera que el rang de transparència no és estrictament cec; el predictor
robust és l'etiqueta objectiva flexiu/derivatiu. La convergència a ~0,60 amb
tall morfèmic és el resultat sòlid; l'absència de diferència flexió/derivació
s'ha de llegir com "no hi ha evidència de diferència", no com "són idèntiques".

## Patrons indoeuropeus (findings §11)

- **Gradient de Sturtevant**: les famílies verbals són petites (15 regulars, 15
  d'alternança, 8 suppletius) i els IC se solapen; el resultat "regular ≳
  irregular" és una tendència feble, no una prova. Els suppletius i les
  alternances **no tenen frontera de morfema** neta, així que només s'hi pot
  llegir la geometria *nadiua*, no un delta morfèmic.
- **Profunditat**: el contrast és d'un sol sufix (`-ció`) sobre bases d'un sol
  esquema (`-itzar`); el resultat (d1 > d0) reflecteix que `-ització` és
  especialment regular, i no s'ha de generalitzar a "més profunditat sempre
  ajuda".
- **Prefixos**: `in-/im-` barreja al·lomorfs (`in|útil` vs `im|possible`); la
  frontera gold segueix la forma de superfície.

## Encuadre dels models del BSC

La família Salamandra / ALIA del BSC s'inclou com a control català-aware i es
descriu de manera **neutra al llarg de tot l'estudi**. Quan el seu
tokenitzador fragmenta menys el català, o quan és el que més guanya amb la
re-segmentació alineada amb els morfemes, això es reporta com a observacions
empíriques, no com a judicis sobre els models ni els seus autors.
