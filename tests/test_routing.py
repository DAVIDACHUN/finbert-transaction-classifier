import numpy as np

from txnclf.data import CATEGORIES, GenConfig, generate
from txnclf.routing import (
    apply_threshold,
    expected_calibration_error,
    fit_temperature,
    pick_threshold,
    route,
    softmax,
)


def test_generator_is_deterministic_and_labelled():
    a = generate(GenConfig(seed=1, n_train=200, n_val=50, n_test_seen=50, n_test_unseen=50))
    b = generate(GenConfig(seed=1, n_train=200, n_val=50, n_test_seen=50, n_test_unseen=50))
    assert a["train"].equals(b["train"])
    assert set(a["train"].label) <= set(CATEGORIES)
    assert a["train"].description.str.len().max() <= 40


def test_unseen_split_uses_held_out_counterparties():
    from txnclf.data import split_stems
    import random

    seen, unseen = split_stems(random.Random(0), 0.25)
    for cat in CATEGORIES:
        assert not set(seen[cat]) & set(unseen[cat])
        assert unseen[cat] and seen[cat]


def test_softmax_rows_sum_to_one():
    p = softmax(np.random.default_rng(0).normal(size=(10, 4)))
    assert np.allclose(p.sum(1), 1.0)


def test_threshold_meets_target_on_fit_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(size=(2000, 3))
    logits[np.arange(2000), y] += rng.uniform(0, 4, 2000)  # varying signal strength
    p = softmax(logits)
    th = pick_threshold(p, y, 0.95)
    r = apply_threshold(p, y, th)
    assert r.auto_accuracy >= 0.95
    assert 0 < r.coverage < 1
    assert abs(r.coverage + r.review_rate - 1) < 1e-9


def test_unreachable_target_routes_everything_to_review():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.4, 0.6], [0.6, 0.4], [0.4, 0.6], [0.6, 0.4]])  # always wrong
    r = apply_threshold(p, y, pick_threshold(p, y, 0.99))
    assert r.coverage == 0.0


def test_temperature_scaling_reduces_ece_for_overconfident_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 3000)
    logits = rng.normal(size=(3000, 4))
    logits[np.arange(3000), y] += 1.0
    overconf = logits * 5
    T = fit_temperature(overconf, y)
    assert T > 1
    assert expected_calibration_error(softmax(overconf, T), y) < expected_calibration_error(softmax(overconf), y)


def test_route_decision():
    out = route(np.array([0.05, 0.95]), ["a", "b"], 0.9)
    assert out == {"label": "b", "confidence": 0.95, "route": "AUTO"}
    assert route(np.array([0.4, 0.6]), ["a", "b"], 0.9)["route"] == "HUMAN_REVIEW"
