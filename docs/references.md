# Referències i treball relacionat

Estudis anteriors sobre els quals es construeix aquest treball, agrupats per
tema. Per a cada bloc s'indica la relació amb el nostre estudi.

---

## 1. Tokenització en subparaules (mètodes)

- **Sennrich, Haddow & Birch (2016).** *Neural Machine Translation of Rare
  Words with Subword Units.* ACL 2016.
  [arXiv:1508.07909](https://arxiv.org/abs/1508.07909) — introdueix el **BPE**,
  l'algorisme darrere de la majoria de tokenitzadors del nostre panell.
- **Kudo (2018).** *Subword Regularization: Improving Neural Network
  Translation Models with Multiple Subword Candidates.* ACL 2018.
  [arXiv:1804.10959](https://arxiv.org/abs/1804.10959) — el model **Unigram**,
  alternativa al BPE.
- **Kudo & Richardson (2018).** *SentencePiece: A simple and language
  independent subword tokenizer and detokenizer for Neural Text Processing.*
  EMNLP 2018 (demos). [arXiv:1808.06226](https://arxiv.org/abs/1808.06226) — la
  implementació (marcador `▁`) que fan servir Gemma i Salamandra; el nostre
  empalmament de *token ids* hi és agnòstic (§4 de la metodologia).

## 2. Tokenització, equitat i cost entre llengües

- **Petrov, La Malfa, Torr & Bibi (2023).** *Language Model Tokenizers
  Introduce Unfairness Between Languages.* NeurIPS 2023.
  [arXiv:2305.15425](https://arxiv.org/abs/2305.15425) — mostra que una mateixa
  frase pot costar fins a 15× més *tokens* segons la llengua. El nostre estudi
  ho instancia en català (~1,7× respecte a l'anglès; fins a ~4× amb el punt
  volat) i hi afegeix la conseqüència geomètrica.
- **Ahia, Kumar, Gonen, Kasai, Mortensen, Smith & Tsvetkov (2023).** *Do All
  Languages Cost the Same? Tokenization in the Era of Commercial Language
  Models.* EMNLP 2023.
  [arXiv:2305.13707](https://arxiv.org/abs/2305.13707) — la no-uniformitat de
  la fertilitat com a problema d'equitat i de cost d'API.
- **Rust, Pfeiffer, Vulić, Ruder & Gurevych (2021).** *How Good is Your
  Tokenizer? On the Monolingual Performance of Multilingual Language Models.*
  ACL 2021. [arXiv:2012.15613](https://arxiv.org/abs/2012.15613) — un
  tokenitzador dedicat a la llengua millora el rendiment; motiva el nostre
  control català-aware (Salamandra/ALIA).

## 3. Tokenització conscient de la morfologia

- **Bostrom & Durrett (2020).** *Byte Pair Encoding is Suboptimal for Language
  Model Pretraining.* Findings of EMNLP 2020.
  [arXiv:2004.03720](https://arxiv.org/abs/2004.03720) — el BPE s'alinea pitjor
  amb la morfologia que l'Unigram; rerefons del nostre **recall de frontera
  morfèmica**.
- **Hofmann, Pierrehumbert & Schütze (2021).** *Superbizarre Is Not Superb:
  Derivational Morphology Improves BERT's Interpretation of Complex Words.*
  ACL-IJCNLP 2021. [arXiv:2101.00403](https://arxiv.org/abs/2101.00403) — una
  segmentació derivativa (DelBERT) millora les representacions semàntiques;
  precedent directe del nostre contrafactual de **segmentació morfèmica**.
- **Hofmann, Schütze & Pierrehumbert (2022).** *An Embarrassingly Simple Method
  to Mitigate Undesirable Properties of Pretrained Language Model Tokenizers
  (FLOTA).* ACL 2022 (short).
  [aclanthology 2022.acl-short.43](https://aclanthology.org/2022.acl-short.43/)
  — preserva l'estructura morfològica reusant el vocabulari existent, com el
  nostre empalmament en temps d'inferència.
- **Jabbar (2023).** *MorphPiece: A Linguistic Tokenizer for Large Language
  Models.* [arXiv:2307.07262](https://arxiv.org/abs/2307.07262) — un
  tokenitzador basat en morfemes; un exemple del que anomenem tokenització
  "universal" conscient dels morfemes.
- **Creutz & Lagus (2007).** *Unsupervised Models for Morpheme Segmentation and
  Morphology Learning.* ACM TSLP — el mètode **Morfessor** original.
- **Virpioja, Smit, Grönroos & Kurimo (2013).** *Morfessor 2.0: Python
  Implementation and Extensions for Morfessor Baseline.* Aalto University,
  Report 25/2013.
  [aaltodoc](https://aaltodoc.aalto.fi/handle/123456789/11836) — la
  implementació que fem servir com a **segmentador realista no supervisat**
  (condició `morfessor`), alternativa pràctica al nostre oracle de fronteres
  gold.

## 4. Geometria de les representacions i subespais lineals

- **Mikolov, Yih & Zweig (2013).** *Linguistic Regularities in Continuous Space
  Word Representations.* NAACL-HLT 2013 — les analogies vectorials
  (*rei − home + dona ≈ reina*); base de la nostra **precisió d'analogia**.
- **Bolukbasi, Chang, Zou, Saligrama & Kalai (2016).** *Man is to Computer
  Programmer as Woman is to Homemaker? Debiasing Word Embeddings.*
  [arXiv:1607.06520](https://arxiv.org/abs/1607.06520) — la idea de
  *direcció* lineal en l'espai (gènere); manllevem el marc per a la
  **"direcció -ment"**.
- **Park, Choe & Veitch (2024).** *The Linear Representation Hypothesis and the
  Geometry of Large Language Models.* ICML 2024.
  [arXiv:2311.03658](https://arxiv.org/abs/2311.03658) — formalitza quan els
  conceptes són direccions lineals; fonament de mesurar la morfologia com a
  direcció consistent.

## 5. Models catalans i multilingües (BSC)

- **Gonzalez-Agirre et al. — Language Technologies Unit, BSC (2025).**
  *Salamandra Technical Report.*
  [arXiv:2502.08489](https://arxiv.org/abs/2502.08489) — la família Salamandra
  (2B/7B) i ALIA (40B), entrenades en 35 llengües europees incloent-hi el
  català; el nostre **control català-aware**. Pesos a
  [huggingface.co/BSC-LT](https://huggingface.co/BSC-LT).

## 6. Context cultural i estudi germà

- **Vinaixa Roselló (2026).** *Coca Is Not Cocaine: Three Lexical-Cultural
  Collision Modes in Open-Weight LLMs, Probed in Catalan.*
  [github.com/xaviviro/coca-is-not-cocaine](https://github.com/xaviviro/coca-is-not-cocaine)
  — estudi germà; en reutilitzem el panell de models i la metodologia de
  bootstrap aparellat.
- **Ada Lovelace Institute (2025).** *Tokenising culture: causes and
  consequences of cultural misalignment in LLMs.*
  [adalovelaceinstitute.org](https://www.adalovelaceinstitute.org/blog/cultural-misalignment-llms/)
  — context divulgatiu sobre el cost cultural de la tokenització.
