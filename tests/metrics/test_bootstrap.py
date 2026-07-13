"""Paired bootstrap: bracketing, shrinking width, and the paired-resampling guard.

Small ``n_resamples`` and fixed seeds keep these fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from conftest import make_calibrated_confidence, make_calibrated_softmax, make_gold, make_softmax
from scipy.stats import bootstrap  # type: ignore[import-untyped]

from frontier.metrics.bootstrap import (
    accuracy_ci,
    ece_ci,
    paired_delta_accuracy_ci,
    paired_delta_ece_ci,
)
from frontier.metrics.calibration import ece_from_confidence, top_label

RESAMPLES = 999
RATE = 0.7


def test_accuracy_ci_point_is_the_sample_accuracy_and_is_bracketed() -> None:
    correct = np.array([True] * 700 + [False] * 300)
    interval = accuracy_ci(correct, n_resamples=RESAMPLES, rng=42)
    assert interval.point == pytest.approx(RATE)
    assert interval.low <= interval.point <= interval.high


def test_accuracy_ci_width_shrinks_with_sample_size() -> None:
    rng = np.random.default_rng(1)
    small = rng.uniform(0.0, 1.0, size=400) < RATE
    large = rng.uniform(0.0, 1.0, size=4000) < RATE
    small_interval = accuracy_ci(small, n_resamples=RESAMPLES, rng=1)
    large_interval = accuracy_ci(large, n_resamples=RESAMPLES, rng=1)
    assert (large_interval.high - large_interval.low) < (small_interval.high - small_interval.low)


def test_accuracy_ci_accepts_a_generator_or_no_seed() -> None:
    correct = make_gold(1000, 2, np.random.default_rng(2)).astype(np.bool_)
    seeded = accuracy_ci(correct, n_resamples=RESAMPLES, rng=np.random.default_rng(0))
    unseeded = accuracy_ci(correct, n_resamples=RESAMPLES)
    assert np.isfinite(seeded.low) and np.isfinite(seeded.high)
    assert unseeded.high - unseeded.low > 0.0


def test_ece_ci_point_matches_and_brackets() -> None:
    rng = np.random.default_rng(4)
    _probs, _gold, confidence, correct = make_calibrated_softmax(3000, rng)
    interval = ece_ci(confidence, correct, n_bins=10, n_resamples=RESAMPLES, rng=3)
    assert interval.point == ece_from_confidence(confidence, correct, n_bins=10)
    assert interval.low <= interval.point <= interval.high


def test_paired_delta_ece_ci_cancels_to_zero_on_identical_variants() -> None:
    rng = np.random.default_rng(3)
    probs = make_softmax(1500, 4, rng, sharpness=1.5)
    gold = make_gold(1500, 4, rng)
    confidence, correct = top_label(probs, gold)
    interval = paired_delta_ece_ci(
        confidence,
        correct,
        confidence.copy(),
        correct.copy(),
        n_bins=10,
        n_resamples=RESAMPLES,
        rng=7,
    )
    assert interval.point == 0.0
    assert interval.low == 0.0
    assert interval.high == 0.0


def test_paired_delta_accuracy_ci_cancels_to_zero_on_identical_variants() -> None:
    rng = np.random.default_rng(3)
    correct = make_gold(1500, 2, rng).astype(np.bool_)
    interval = paired_delta_accuracy_ci(correct, correct.copy(), n_resamples=RESAMPLES, rng=7)
    assert interval.point == 0.0
    assert interval.low == 0.0
    assert interval.high == 0.0


def test_unpaired_resampling_widens_the_identical_variant_interval() -> None:
    rng = np.random.default_rng(3)
    correct = make_gold(1500, 2, rng).astype(np.bool_)

    def delta(left: npt.NDArray[np.bool_], right: npt.NDArray[np.bool_]) -> float:
        return float(np.mean(right) - np.mean(left))

    result = bootstrap(
        (correct, correct.copy()),
        delta,
        paired=False,
        vectorized=False,
        n_resamples=RESAMPLES,
        method="percentile",
        rng=np.random.default_rng(7),
    )
    interval = result.confidence_interval
    assert interval.high - interval.low > 0.0


def test_paired_delta_ece_ci_between_independent_calibrated_draws_contains_zero() -> None:
    confidence_a, correct_a = make_calibrated_confidence(2000, np.random.default_rng(10))
    confidence_b, correct_b = make_calibrated_confidence(2000, np.random.default_rng(11))
    interval = paired_delta_ece_ci(
        confidence_a, correct_a, confidence_b, correct_b, n_bins=10, n_resamples=RESAMPLES, rng=5
    )
    assert interval.low <= 0.0 <= interval.high
