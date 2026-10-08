"""TF-IDF (char n-grams) + logistic regression baseline.

A strong, cheap baseline is the honest yardstick for any transformer: if the
transformer can't beat it, it isn't earning its compute.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from txnclf.data import CATEGORIES, to_model_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts/baseline")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lab2id = {c: i for i, c in enumerate(CATEGORIES)}

    splits = {s: pd.read_csv(data / f"{s}.csv") for s in ["train", "val", "test_seen", "test_unseen"]}
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, min_df=2, sublinear_tf=True),
        LogisticRegression(C=10.0, max_iter=3000),
    )
    t0 = time.time()
    pipe.fit(to_model_text(splits["train"]), splits["train"].label.map(lab2id))
    print(f"baseline fit in {time.time() - t0:.1f}s")

    for name, df in splits.items():
        if name == "train":
            continue
        # LR decision_function gives logits in CATEGORIES order (classes are 0..11)
        logits = pipe.decision_function(to_model_text(df))
        y = df.label.map(lab2id).to_numpy()
        np.savez(out / f"{name}.npz", logits=logits, y=y)
        print(f"{name:12s} acc={np.mean(logits.argmax(1) == y):.4f}")


if __name__ == "__main__":
    main()
