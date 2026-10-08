"""Classify + route ad-hoc transactions with the fine-tuned model.

    python -m txnclf.predict --direction DEBIT --amount 1450 "ACH DEBIT SBA LOAN PMT 0042"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from txnclf.data import CATEGORIES
from txnclf.routing import route, softmax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("description", nargs="+")
    ap.add_argument("--direction", choices=["DEBIT", "CREDIT"], default="DEBIT")
    ap.add_argument("--amount", type=float, default=100.0)
    ap.add_argument("--model-dir", default="artifacts/finbert/model")
    ap.add_argument("--metrics", default="reports/metrics.json")
    ap.add_argument("--target", default="0.99", help="routing target used to pick the threshold")
    args = ap.parse_args()

    m = json.loads(Path(args.metrics).read_text())["finbert"]
    T, threshold = m["temperature"], m["thresholds"][args.target]
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval()

    texts = [f"{args.direction} {args.amount:.0f} | {d}" for d in args.description]
    with torch.no_grad():
        logits = model(**tok(texts, return_tensors="pt", padding=True)).logits.numpy()
    for text, p in zip(texts, softmax(logits, T)):
        print(json.dumps({"input": text, **route(np.asarray(p), CATEGORIES, threshold)}))


if __name__ == "__main__":
    main()
