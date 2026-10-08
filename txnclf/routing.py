"""Confidence-threshold auto-routing, calibration and coverage/accuracy curves.

Routing rule: a prediction is AUTO-ROUTED when its top-class probability is
>= threshold; otherwise it goes to the HUMAN REVIEW queue. The threshold is
chosen on the validation split as the *lowest* confidence that still keeps
auto-routed accuracy >= a target, and is then frozen and applied to test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray, grid=None) -> float:
    """Temperature scaling (Guo et al., 2017) by NLL grid search on validation."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200)) if grid is None else grid
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = softmax(logits, t)
        nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def coverage_accuracy_curve(probs: np.ndarray, y: np.ndarray, n_points: int = 101) -> dict:
    """Accuracy of the auto-routed subset as a function of coverage."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    order = np.argsort(-conf)
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(y) + 1)
    cov = np.linspace(0.01, 1.0, n_points)
    idx = np.clip((cov * len(y)).astype(int) - 1, 0, len(y) - 1)
    return {"coverage": cov.tolist(), "accuracy": cum_acc[idx].tolist()}


@dataclass
class RoutingResult:
    threshold: float
    coverage: float          # share auto-routed
    auto_accuracy: float     # accuracy on auto-routed rows
    review_rate: float       # share sent to humans
    review_accuracy: float   # model accuracy on the rows humans will see
    overall_accuracy: float  # model accuracy on everything (no routing)


def pick_threshold(probs: np.ndarray, y: np.ndarray, target_accuracy: float) -> float:
    """Lowest threshold whose auto-routed accuracy >= target (on validation)."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    order = np.argsort(-conf)
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(y) + 1)
    ok = np.where(cum_acc >= target_accuracy)[0]
    if len(ok) == 0:
        return 1.0 + 1e-9  # nothing qualifies -> route everything to review
    k = ok.max()
    return float(conf[order][k])


def apply_threshold(probs: np.ndarray, y: np.ndarray, threshold: float) -> RoutingResult:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    auto = conf >= threshold
    return RoutingResult(
        threshold=float(threshold),
        coverage=float(auto.mean()),
        auto_accuracy=float(correct[auto].mean()) if auto.any() else float("nan"),
        review_rate=float((~auto).mean()),
        review_accuracy=float(correct[~auto].mean()) if (~auto).any() else float("nan"),
        overall_accuracy=float(correct.mean()),
    )


def route(prob_row: np.ndarray, labels: list[str], threshold: float) -> dict:
    k = int(prob_row.argmax())
    conf = float(prob_row[k])
    return {
        "label": labels[k],
        "confidence": round(conf, 4),
        "route": "AUTO" if conf >= threshold else "HUMAN_REVIEW",
    }
