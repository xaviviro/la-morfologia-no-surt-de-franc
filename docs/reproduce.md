# Reproducció

Tot és determinista i corre en local — el tram de geometria cap en una GPU de
16 GB a bf16; l'auditoria de tokenitzadors només necessita els fitxers del
tokenitzador (sense pesos, sense GPU).

## 0. Entorn

```bash
cd tokenizer-analisis
uv sync
uv run pytest -q          # tots els tests han de passar
```

Alguns models del panell són *gated* a Hugging Face. Si un tokenitzador o
model no es pot baixar, executeu `hf auth login` (i accepteu les llicències
dels models) i torneu-ho a provar; l'script d'auditoria fa SKIP de qualsevol
tokenitzador que no pugui carregar i ho reporta.

## 1. Construir el lèxic

```bash
uv run python scripts/m01_build_lexicon.py
# -> data/morph_pairs.csv  (319 files, 11 famílies)
```

## 2. Auditoria del tokenitzador (RQ1) — els 11 tokenitzadors, sense GPU

```bash
uv run python scripts/m02_tokenize_audit.py
# -> out/tokenize_audit.csv  (11 models x 319 paraules)
```

## 3. Extreure embeddings nadius + morfèmics (RQ2/RQ3) — 5 models petits

```bash
for M in google/gemma-2-2b google/gemma-4-E2B Qwen/Qwen2-1.5B \
         Qwen/Qwen3.5-4B-Base BSC-LT/salamandra-2b; do
  uv run python scripts/m03_extract.py --model "$M"
done
# -> out/{slug}/embeddings_layer{L}.npz + metadata.parquet  (per model)
```

Cada model escriu vectors per capa amb 4 condicions de segmentació × 2 rols
(base/derivat): **nadiu**, **morfèmic** (oracle), **aleatori** (placebo) i
**Morfessor** (segmentador realista no supervisat, entrenat al vol sobre les
paraules del lèxic). `m03` entrena Morfessor automàticament; per generar només
la taula de concordança Morfessor↔gold pots executar
`uv run python scripts/morf_seg.py`.

## 4. Mètriques de geometria (RQ2/RQ3)

```bash
uv run python scripts/m04_geometry.py
# -> out/geometry_metrics.csv  (5 models x 3 capes x 11 famílies x {nadiu,morfèmic,delta})
```

## 5. Figures

```bash
uv run python scripts/m07_regularity.py   # regularitat: flexió vs derivació
uv run python scripts/m05_figs.py
uv run python scripts/m06_figs.py   # figures explicatives extra
# -> out/figs/*.png
```

## El pipeline d'un cop d'ull

| script | entrada | sortida | cal GPU |
| --- | --- | --- | --- |
| `m01_build_lexicon.py` | (llistes curades) | `data/morph_pairs.csv` | no |
| `m02_tokenize_audit.py` | lèxic | `out/tokenize_audit.csv` | no |
| `m03_extract.py` | lèxic | `out/{slug}/*.npz` + `metadata.parquet` | sí |
| `m04_geometry.py` | npz + metadata | `out/geometry_metrics.csv`, `out/geometry_aggregate_ci.csv` | no |
| `m05_figs.py` | auditoria + mètriques + npz | `out/figs/*.png` | no |
| `m06_figs.py` | auditoria + mètriques + tokenitzadors | `out/figs/*.png` (extra) | no |
| `morf_seg.py` | lèxic | `out/morfessor_agreement.csv` | no |
| `m07_regularity.py` | mètriques + `data/family_traits.csv` | `out/regularity_*.csv` | no |

La lògica pura viu a `scripts/geom_lib.py` (mètriques de geometria) i
`scripts/embed_lib.py` (ajudes de tokenització + extracció), totes dues
cobertes per `tests/`. Les etiquetes en català dels gràfics són a
`scripts/labels_ca.py`.
