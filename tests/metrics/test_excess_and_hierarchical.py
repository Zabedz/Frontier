"""Excess damage, the AUROC delta, and the seed-and-question interval."""

from __future__ import annotations

import numpy as np
import pytest

from frontier.metrics._array import CorrectArray, FloatArray
from frontier.metrics.bootstrap import (
    paired_seeded_ci,
    paired_seeded_hierarchical_ci,
    seeded_quantity,
)
from frontier.metrics.calibration import ece_from_confidence
from frontier.metrics.discrimination import auroc

N_ITEMS = 2000
RESAMPLES = 199
Row = tuple[FloatArray, CorrectArray]


def _overconfident(seed: int, *, accuracy: float, lift: float) -> Row:
    """Confidence above accuracy in every bin: accuracy ~0.6, confidence 0.7 to 1."""
    rng = np.random.default_rng(seed)
    confidence: FloatArray = np.clip(rng.uniform(0.85, 0.99, N_ITEMS) + lift, 0.0, 1.0)
    correct: CorrectArray = rng.random(N_ITEMS) < accuracy
    return confidence, correct


def test_excess_damage_is_the_ece_rise_plus_the_accuracy_change() -> None:
    reference = _overconfident(0, accuracy=0.65, lift=0.0)
    variant = _overconfident(1, accuracy=0.6, lift=0.0)
    expected = (ece_from_confidence(*variant) - ece_from_confidence(*reference)) + (
        float(np.mean(variant[1])) - float(np.mean(reference[1]))
    )
    assert seeded_quantity("excess_damage", reference, [variant]) == pytest.approx(expected)


def test_excess_damage_equals_the_confidence_shift_when_every_bin_is_overconfident() -> None:
    reference = _overconfident(0, accuracy=0.65, lift=0.0)
    seeds = [_overconfident(seed, accuracy=0.6, lift=-0.01 * seed) for seed in (1, 2, 3)]
    excess = seeded_quantity("excess_damage", reference, seeds)
    shift = seeded_quantity("delta_confidence", reference, seeds)
    assert excess == pytest.approx(shift, abs=1e-12)


def test_a_pure_accuracy_loss_leaves_no_excess_damage() -> None:
    confidence, correct = _overconfident(0, accuracy=0.65, lift=0.0)
    flipped = correct.copy()
    flipped[np.flatnonzero(correct)[:100]] = False
    assert seeded_quantity("excess_damage", (confidence, correct), [(confidence, flipped)]) == (
        pytest.approx(0.0, abs=1e-12)
    )
    assert seeded_quantity("damage_gap", (confidence, correct), [(confidence, flipped)]) > 0.0


def test_auroc_delta_averages_the_seed_aurocs() -> None:
    reference = _overconfident(0, accuracy=0.65, lift=0.0)
    seeds = [_overconfident(seed, accuracy=0.6, lift=0.0) for seed in (1, 2)]
    expected = float(np.mean([auroc(*seed) for seed in seeds])) - auroc(*reference)
    assert seeded_quantity("delta_auroc", reference, seeds) == pytest.approx(expected)


def test_the_hierarchical_interval_widens_when_the_seeds_disagree() -> None:
    reference = _overconfident(0, accuracy=0.65, lift=0.0)
    agreeing = [_overconfident(1, accuracy=0.6, lift=-0.05)] * 3
    disagreeing = [_overconfident(1, accuracy=0.6, lift=lift) for lift in (-0.12, -0.05, +0.02)]

    def width(seeds: list[Row], hierarchical: bool) -> float:
        fn = paired_seeded_hierarchical_ci if hierarchical else paired_seeded_ci
        interval = fn(reference, seeds, quantity="delta_confidence", n_resamples=RESAMPLES, rng=0)
        return interval.high - interval.low

    assert width(agreeing, True) == pytest.approx(width(agreeing, False), rel=0.3)
    assert width(disagreeing, True) > 3 * width(disagreeing, False)


def test_the_hierarchical_interval_is_reproducible_and_centred_on_the_point() -> None:
    reference = _overconfident(0, accuracy=0.65, lift=0.0)
    seeds = [_overconfident(seed, accuracy=0.6, lift=-0.02 * seed) for seed in (1, 2, 3)]
    first = paired_seeded_hierarchical_ci(
        reference, seeds, quantity="excess_damage", n_resamples=RESAMPLES, rng=7
    )
    again = paired_seeded_hierarchical_ci(
        reference, seeds, quantity="excess_damage", n_resamples=RESAMPLES, rng=7
    )
    assert (first.low, first.point, first.high) == (again.low, again.point, again.high)
    assert first.low <= first.point <= first.high


def test_the_hierarchical_interval_refuses_no_seeds() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        paired_seeded_hierarchical_ci(
            _overconfident(0, accuracy=0.6, lift=0.0),
            [],
            quantity="delta_confidence",
            n_resamples=RESAMPLES,
        )
