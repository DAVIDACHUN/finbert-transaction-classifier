"""Build the results tables and plots from saved logits.

Reads artifacts/<model>/{val,test_seen,test_unseen}.npz, fits temperature on
val, picks routing thresholds on val, and evaluates on both test splits.
Writes reports/metrics.json, reports/results.md and PNG charts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from txnclf.data import CATEGORIES
from txnclf.routing import (
    apply_threshold,
    coverage_accuracy_curve,
    expected_calibration_error,
    fit_temperature,
    pick_threshold,
    softmax,
)

TARGETS = [0.98, 0.99, 0.995]
MODELS = {"baseline": "TF-IDF + LogReg", "finbert": "FinBERT (fine-tuned)"}
COLORS = {"baseline": "#8a8f98", "finbert": "#1f4e79"}


def load(art: Path, model: str, split: str):
    z = np.load(art / model / f"{split}.npz")
    return z["logits"], z["y"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    art, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    curves: dict = {}
    for m in MODELS:
        if not (art / m / "val.npz").exists():
            print(f"skip {m}: no artifacts")
            continue
        lv, yv = load(art, m, "val")
        T = fit_temperature(lv, yv)
        pv = softmax(lv, T)
        r = {"temperature": T, "splits": {}}
        thresholds = {t: pick_threshold(pv, yv, t) for t in TARGETS}
        r["thresholds"] = {str(k): v for k, v in thresholds.items()}
        for split in ["test_seen", "test_unseen"]:
            lt, yt = load(art, m, split)
            p_raw, p = softmax(lt, 1.0), softmax(lt, T)
            s = {
                "accuracy": float((p.argmax(1) == yt).mean()),
                "macro_f1": float(f1_score(yt, p.argmax(1), average="macro")),
                "ece_raw": expected_calibration_error(p_raw, yt),
                "ece_temp_scaled": expected_calibration_error(p, yt),
                "routing": {str(t): apply_threshold(p, yt, th).__dict__ for t, th in thresholds.items()},
            }
            r["splits"][split] = s
            curves[(m, split)] = coverage_accuracy_curve(p, yt)
            if m == "finbert" or not (art / "finbert" / "val.npz").exists():
                cm = confusion_matrix(yt, p.argmax(1), labels=range(len(CATEGORIES)))
                plot_confusion(cm, out / f"confusion_{m}_{split}.png", f"{MODELS[m]} - {split}")
        results[m] = r
        meta = art / m / "train_meta.json"
        if meta.exists():
            r["train_meta"] = json.loads(meta.read_text())

    results["finbert_seeds"] = seed_table(art)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_curves(curves, out / "coverage_vs_accuracy.png")
    (out / "results.md").write_text(markdown(results))
    print((out / "results.md").read_text())


def seed_table(art: Path) -> dict:
    """Accuracy across fine-tuning seeds (artifacts/finbert = seed 42, finbert_s<N> = others)."""
    runs = {}
    for d in sorted([art / "finbert", *art.glob("finbert_s*")]):
        if (d / "test_seen.npz").exists():
            runs[d.name] = {s: float((lambda z: (z["logits"].argmax(1) == z["y"]).mean())(np.load(d / f"{s}.npz")))
                            for s in ["val", "test_seen", "test_unseen"]}
    if not runs:
        return {}
    summary = {s: {"mean": float(np.mean([r[s] for r in runs.values()])),
                   "std": float(np.std([r[s] for r in runs.values()], ddof=1)) if len(runs) > 1 else 0.0}
               for s in ["val", "test_seen", "test_unseen"]}
    return {"runs": runs, "summary": summary}


def plot_curves(curves: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, split in zip(axes, ["test_seen", "test_unseen"]):
        for m in MODELS:
            if (m, split) in curves:
                c = curves[(m, split)]
                ax.plot(np.array(c["coverage"]) * 100, np.array(c["accuracy"]) * 100,
                        label=MODELS[m], color=COLORS[m], lw=2)
        ax.set_title({"test_seen": "Seen counterparties", "test_unseen": "Unseen counterparties"}[split])
        ax.set_xlabel("Coverage: % of transactions auto-routed")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy on auto-routed set (%)")
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("Coverage vs. accuracy (sorted by calibrated confidence) - synthetic data")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(CATEGORIES)), CATEGORIES, fontsize=8)
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def markdown(results: dict) -> str:
    seeds = results.get("finbert_seeds", {})
    results = {k: v for k, v in results.items() if k in MODELS}
    lines = ["### Headline metrics (test splits)", "",
             "| Model | Split | Accuracy | Macro-F1 | ECE raw | ECE temp-scaled |",
             "|---|---|---|---|---|---|"]
    for m, r in results.items():
        for split, s in r["splits"].items():
            lines.append(f"| {MODELS[m]} | {split} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | "
                         f"{s['ece_raw']:.3f} | {s['ece_temp_scaled']:.3f} |")
    lines += ["", "### Auto-routing (threshold picked on validation, applied to test)", "",
              "| Model | Split | Target acc (val) | Threshold | Coverage (auto) | Auto-routed acc | Human-review rate |",
              "|---|---|---|---|---|---|---|"]
    for m, r in results.items():
        for split, s in r["splits"].items():
            for t, rr in s["routing"].items():
                lines.append(f"| {MODELS[m]} | {split} | {float(t):.1%} | {rr['threshold']:.3f} | "
                             f"{rr['coverage']:.1%} | {rr['auto_accuracy']:.2%} | {rr['review_rate']:.1%} |")
    if seeds:
        lines += ["", f"### FinBERT seed variance ({len(seeds['runs'])} fine-tuning runs)", "",
                  "| Split | Mean accuracy | Std dev |", "|---|---|---|"]
        for s, v in seeds["summary"].items():
            lines.append(f"| {s} | {v['mean']:.3f} | {v['std']:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
