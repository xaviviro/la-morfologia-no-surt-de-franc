"""Continued-pretraining (full FT) of one model on one condition's token stream.

Saves training logs (loss/LR/grad-norm/eval) and, if HF_TOKEN is set, pushes the
trained weights to a PRIVATE HF repo with a Catalan model card linking the study.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from part2.config import SEED, SEQ_LEN
from scripts.embed_lib import slugify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "part2" / "out"
STUDY_URL = "https://github.com/xaviviro/la-morfologia-no-surt-de-franc"
COND_CA = {"A": "nadiua (control)", "B": "morfèmica (segmentador de regles català)"}


class PackedDataset(Dataset):
    def __init__(self, ids: np.ndarray, seq_len: int):
        n = (len(ids) // seq_len) * seq_len
        self.blocks = ids[:n].reshape(-1, seq_len)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        x = torch.tensor(self.blocks[i], dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}


def model_card_ca(model_id: str, condition: str, meta: dict) -> str:
    """Catalan model card (README.md) for the uploaded weights."""
    cond_desc = COND_CA[condition]
    seed = meta.get("seed", SEED)
    return f"""---
language: ca
license: apache-2.0
tags: [catalan, morphology, tokenization, continued-pretraining, research]
base_model: {model_id}
---

# {slugify(model_id)} — continued-pretraining {condition} (segmentació {cond_desc}, llavor {seed})

> ⚠️ **Artefacte de recerca**, no un model de producció. Part 2 de l'estudi
> [*La morfologia no surt de franc*]({STUDY_URL}).

Aquest model és **`{model_id}`** sotmès a un *continued-pretraining* controlat
sobre un subconjunt del corpus català **`projecte-aina/CATalog`**, amb la
condició de segmentació **{condition}**:

- **A — nadiua:** tokenització estàndard del model (control).
- **B — morfèmica:** cada paraula es pre-segmenta pels morfemes amb un
  segmentador de regles català (`scripts/rule_seg.py`) i s'empalma amb el mateix
  vocabulari.

**A i B són idèntics en tota la resta** (corpus, passos, LR); per a aquesta
versió s'han executat múltiples llavors per donar potència estadística al test
pareat dins de cada model. L'única variable de contrast és la segmentació.

## Corbes d'entrenament

![Corbes d'entrenament d'aquest run (train loss · LR · grad-norm · eval loss)](train_curves.png)

## Detalls d'entrenament

| | |
| --- | --- |
| model base | `{model_id}` |
| condició | {condition} — {cond_desc} |
| corpus | projecte-aina/CATalog (subconjunt) |
| tokens d'entrenament | {meta.get('n_train_tokens', 'n/d')} |
| èpoques | {meta.get('epochs', 'n/d')} |
| learning rate | {meta.get('lr', 'n/d')} |
| optimitzador | {meta.get('optim', 'adamw_bnb_8bit')} (els pesos es mantenen en bf16) |
| perplexitat final (eval) | {meta.get('final_perplexity', float('nan')):.2f} |
| llavor | {seed} |

## Estudi i reproducció

- Codi, metodologia i resultats: [{STUDY_URL}]({STUDY_URL})
- Llegeix-ne el `README.md` (en català) per al context complet
  (Part 1: geometria; Part 2: aquest reentrenament; Part 3: visió IE).

## Citació

```bibtex
@misc{{vinaixa2026morfologia,
  title        = {{La morfologia no surt de franc: com la tokenitzaci\\'o en
                  subparaules fractura la morfologia catalana}},
  author       = {{Vinaixa Rosell\\'o, Xavier and Font Esp\\'i, Mar\\c{{c}}al}},
  year         = {{2026}},
  institution  = {{Sorensen AI, Barcelona}},
  note         = {{ORCID: 0009-0005-2769-9215}},
  url          = {{{STUDY_URL}}}
}}
```
"""


def _plot_curves(log_csv: Path, out_png: Path, title: str) -> None:
    """4-panel diagnostic plot for one training run: train loss · LR · grad-norm · eval loss."""
    import matplotlib.pyplot as plt
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [("loss", "Train loss", "step"),
              ("learning_rate", "Learning rate", "step"),
              ("grad_norm", "Grad norm", "step"),
              ("eval_loss", "Eval loss", "step")]
    for ax, (col, ttl, xlab) in zip(axes.flatten(), panels):
        if col in df.columns:
            sub = df[df[col].notna()]
            style = "o-" if col == "eval_loss" else "-"
            ax.plot(sub["step"], sub[col], style, lw=1.2)
        ax.set_title(ttl)
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def push_weights(out_dir: Path, model_id: str, condition: str, meta: dict) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set — skipping upload")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    user = api.whoami()["name"]
    seed = meta.get("seed", SEED)
    repo_id = f"{user}/morfo-part2-{slugify(model_id)}-{condition}-s{seed}"
    api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
    (out_dir / "README.md").write_text(model_card_ca(model_id, condition, meta))
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="model")
    print(f"pushed PRIVATE weights + Catalan card -> https://huggingface.co/{repo_id}")


def train(model_id: str, condition: str, seed: int = SEED,
          epochs: float = 1.0, lr: float = 2e-5) -> None:
    slug = slugify(model_id)
    out_dir = OUT / f"{slug}_{condition}_s{seed}"
    # Resumability: a previously-successful run leaves train_meta.json with a finite
    # final_perplexity. Skip it so re-launching the orchestrator doesn't redo work.
    meta_path = out_dir / "train_meta.json"
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
            ppl = prev.get("final_perplexity")
            if ppl is not None and not math.isnan(ppl) and not math.isinf(ppl):
                print(f"skipping {out_dir.name}: already trained (ppl={ppl:.2f})")
                return
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    ids = np.load(OUT / slug / f"corpus_{condition}.npy")
    n_eval = max(SEQ_LEN * 4, len(ids) // 100)
    train_ds = PackedDataset(ids[:-n_eval], SEQ_LEN)
    eval_ds = PackedDataset(ids[-n_eval:], SEQ_LEN)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # Effective batch is fixed at 16 (micro-batch x accumulation) for every model so
    # the optimisation is identical across runs; only the micro-batch (a throughput/
    # memory knob, not a scientific one) shrinks for the larger 4B model.
    micro_bs = 4 if "4B" in model_id else 8
    accum = 16 // micro_bs
    # gemma-4 E2B's parameter shapes crash the bitsandbytes 8-bit optimiser kernel
    # ("invalid configuration argument" in ops.cu); fall back to memory-light Adafactor
    # for that model only. A and B share it, so the within-model A/B test still holds.
    optim = "adafactor" if "E2B" in model_id else "adamw_bnb_8bit"
    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=epochs,
        per_device_train_batch_size=micro_bs, gradient_accumulation_steps=accum,
        per_device_eval_batch_size=micro_bs, learning_rate=lr, lr_scheduler_type="cosine",
        warmup_ratio=0.03, bf16=True, tf32=True, logging_steps=20, eval_strategy="steps",
        eval_steps=400, save_strategy="no", optim=optim, seed=seed, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    pd.DataFrame(trainer.state.log_history).to_csv(out_dir / "train_log.csv", index=False)
    _plot_curves(out_dir / "train_log.csv", out_dir / "train_curves.png",
                 title=f"{slug} · {condition} · llavor {seed}")
    fe = trainer.evaluate()
    meta = {"model": model_id, "condition": condition, "seed": seed,
            "epochs": epochs, "lr": lr, "optim": optim,
            "final_eval_loss": float(fe.get("eval_loss", float("nan"))),
            "final_perplexity": float(np.exp(fe.get("eval_loss", float("nan")))),
            "n_train_tokens": int(len(ids) - n_eval), "seq_len": SEQ_LEN}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out_dir}  final ppl={meta['final_perplexity']:.2f}")
    push_weights(out_dir, model_id, condition, meta)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--condition", required=True, choices=["A", "B"])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=float, default=1.0)
    a = p.parse_args()
    train(a.model, a.condition, seed=a.seed, epochs=a.epochs)


if __name__ == "__main__":
    main()
