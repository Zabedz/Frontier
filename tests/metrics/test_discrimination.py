"""AUROC of confidence for correctness: the sklearn oracle, ties, and the degenerate cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

from frontier.metrics.discrimination import auroc

N_ITEMS = 500
HIT_RATE = 0.6
CHANCE = 0.5  # AUROC of a ranking that carries no information


def test_matches_sklearn_on_continuous_and_tied_confidence() -> None:
    rng = np.random.default_rng(0)
    correct = rng.random(N_ITEMS) < HIT_RATE
    continuous = np.clip(0.5 + 0.3 * correct + rng.normal(0, 0.2, N_ITEMS), 0, 1)
    tied = np.round(continuous, 1)
    for confidence in (continuous, tied):
        assert auroc(confidence, correct) == pytest.approx(roc_auc_score(correct, confidence))


def test_perfect_and_reversed_ranking() -> None:
    correct = np.array([True, True, False, False])
    assert auroc(np.array([0.9, 0.8, 0.3, 0.2]), correct) == 1.0
    assert auroc(np.array([0.2, 0.3, 0.8, 0.9]), correct) == 0.0


def test_constant_confidence_cannot_rank() -> None:
    assert auroc(np.full(4, 0.7), np.array([True, False, True, False])) == CHANCE


def test_one_class_leaves_it_undefined() -> None:
    assert math.isnan(auroc(np.array([0.4, 0.9]), np.array([True, True])))
