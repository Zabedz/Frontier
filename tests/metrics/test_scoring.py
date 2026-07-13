"""Proper scores and the Murphy decomposition reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_discrete_softmax, make_gold, make_softmax

from frontier.metrics.scoring import brier_decomposition, brier_score, nll


def test_brier_score_bounds() -> None:
    rng = np.random.default_rng(0)
    gold = make_gold(400, 4, rng)
    correct = np.zeros((400, 4), dtype=np.float64)
    correct[np.arange(400), gold] = 1.0
    wrong = np.zeros((400, 4), dtype=np.float64)
    wrong[np.arange(400), (gold + 1) % 4] = 1.0
    assert brier_score(correct, gold) == 0.0
    assert brier_score(wrong, gold) == pytest.approx(2.0)


def test_nll_hand_fixture() -> None:
    probs = np.array([[0.5, 0.5], [0.75, 0.25]])
    gold = np.array([0, 1], dtype=np.intp)
    expected = float(np.mean(-np.log(np.array([0.5, 0.25]))))
    assert nll(probs, gold) == pytest.approx(expected)


def test_nll_is_finite_when_gold_probability_is_zero() -> None:
    rng = np.random.default_rng(1)
    gold = make_gold(200, 4, rng)
    wrong = np.zeros((200, 4), dtype=np.float64)
    wrong[np.arange(200), (gold + 1) % 4] = 1.0
    eps = 1e-12
    value = nll(wrong, gold, eps=eps)
    assert np.isfinite(value)
    assert value == pytest.approx(-float(np.log(eps)))


def test_decomposition_reconstructs_binned_brier_continuous() -> None:
    rng = np.random.default_rng(7)
    probs = make_softmax(2000, 4, rng)
    gold = make_gold(2000, 4, rng)
    decomposition = brier_decomposition(probs, gold, n_bins=10)
    reconstructed = decomposition.reliability - decomposition.resolution + decomposition.uncertainty
    assert decomposition.total == pytest.approx(reconstructed, abs=1e-12)


def test_decomposition_reconstructs_raw_brier_on_discrete_forecasts() -> None:
    rng = np.random.default_rng(3)
    probs, gold = make_discrete_softmax(1500, rng)
    decomposition = brier_decomposition(probs, gold, n_bins=10)
    assert decomposition.total == pytest.approx(brier_score(probs, gold), abs=1e-12)


def test_uncertainty_is_one_minus_sum_of_squared_base_rates() -> None:
    rng = np.random.default_rng(5)
    gold = np.array([0] * 500 + [1] * 300 + [2] * 200, dtype=np.intp)
    probs = make_softmax(1000, 3, rng)
    decomposition = brier_decomposition(probs, gold, n_bins=10)
    base_rate = np.array([0.5, 0.3, 0.2])
    assert decomposition.uncertainty == pytest.approx(1.0 - float(np.sum(base_rate**2)), abs=1e-12)
