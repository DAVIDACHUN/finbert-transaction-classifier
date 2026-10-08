"""Fine-tune a Hugging Face encoder (default: ProsusAI/finbert) for 12-way
transaction categorisation.

FinBERT ships with a 3-class sentiment head; we discard it and attach a fresh
12-class head (``ignore_mismatched_sizes=True``). Sequences are short bank
descriptors, so ``max_length=48`` is plenty and training is CPU/Apple-MPS
feasible (see README for timings).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from txnclf.data import CATEGORIES, to_model_text

SPLITS = ["train", "val", "test_seen", "test_unseen"]


def build_ds(df: pd.DataFrame, tok, max_len: int) -> Dataset:
    lab2id = {c: i for i, c in enumerate(CATEGORIES)}
    ds = Dataset.from_dict({"text": to_model_text(df), "label": df.label.map(lab2id).tolist()})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True, remove_columns=["text"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ProsusAI/finbert")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts/finbert")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(CATEGORIES),
        id2label=dict(enumerate(CATEGORIES)),
        label2id={c: i for i, c in enumerate(CATEGORIES)},
        ignore_mismatched_sizes=True,
    )
    dfs = {s: pd.read_csv(data / f"{s}.csv") for s in SPLITS}
    ds = {s: build_ds(df, tok, args.max_len) for s, df in dfs.items()}

    def metrics(p):
        return {"accuracy": float((p.predictions.argmax(-1) == p.label_ids).mean())}

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=128,
        warmup_steps=int(0.06 * args.epochs * -(-len(ds["train"]) // args.batch)),  # 6% linear warmup
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
    )
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    for s in ["val", "test_seen", "test_unseen"]:
        pred = trainer.predict(ds[s])
        np.savez(out / f"{s}.npz", logits=pred.predictions, y=pred.label_ids)
        print(f"{s:12s} acc={np.mean(pred.predictions.argmax(-1) == pred.label_ids):.4f}")

    trainer.save_model(str(out / "model"))
    tok.save_pretrained(str(out / "model"))
    meta = {"base_model": args.model, "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
            "max_len": args.max_len, "device": device, "train_seconds": round(train_secs, 1),
            "n_params": sum(p.numel() for p in model.parameters())}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
