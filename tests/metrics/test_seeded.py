"""The seed-averaged paired statistics: equal to the one-seed functions at one seed, and an
average of per-seed scores at several."""

from __future__ import annotations

import math

import numpy as np
import pytest

from frontier.metrics._array import CorrectArray, FloatArray, LabelArray, ProbMatrix
from frontier.metrics.bootstrap import (
    ScoredItems,
    SeededQuantity,
    paired_damage_gap_ci,
    paired_damage_ratio_ci,
    paired_delta_accuracy_ci,
    paired_delta_confidence_ci,
    paired_delta_ece_ci,
    paired_residual_ece_ci,
    paired_seeded_ci,
    paired_seeded_ratio_ci,
    paired_seeded_residual_ece_ci,
    seeded_quantity,
)
from frontier.metrics.calibration import ece_from_confidence

N_ITEMS = 300
N_FIT = 120
N_OPTIONS = 4
RESAMPLES = 199
Row = tuple[FloatArray, CorrectArray]


def _row(seed: int, *, accuracy: float = 0.65) -> Row:
    rng = np.random.default_rng(seed)
    confidence: FloatArray = rng.uniform(0.6, 0.99, N_ITEMS)
    correct: CorrectArray = rng.random(N_ITEMS) < accuracy
    return confidence, correct


def _scored(seed: int, n_items: int) -> ScoredItems:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, (n_items, N_OPTIONS))
    probs: ProbMatrix = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    gold: LabelArray = np.asarray([rng.choice(N_OPTIONS, p=row) for row in probs], dtype=np.intp)
    sharpened: ProbMatrix = probs**3 / (probs**3).sum(axis=1, keepdims=True)
    return ScoredItems(sharpened, gold, np.full(n_items, N_OPTIONS, dtype=np.intp))


def _same(left: object, right: object) -> None:
    """Bit-identical, a nan bound matching only a nan bound."""
    for name in ("point", "low", "high"):
        a, b = getattr(left, name), getattr(right, name)
        assert a == b or (math.isnan(a) and math.isnan(b)), name


def test_one_seed_matches_the_existing_four_array_statistics() -> None:
    reference, variant = _row(0), _row(1, accuracy=0.6)
    arrays = (*reference, *variant)
    _same(
        paired_seeded_ci(reference, [variant], quantity="delta_ece", n_resamples=RESAMPLES, rng=0),
        paired_delta_ece_ci(*arrays, n_resamples=RESAMPLES, rng=0),
    )
    _same(
        paired_seeded_ci(reference, [variant], quantity="damage_gap", n_resamples=RESAMPLES, rng=0),
        paired_damage_gap_ci(*arrays, n_resamples=RESAMPLES, rng=0),
    )
    seeded_ratio = paired_seeded_ratio_ci(reference, [variant], n_resamples=RESAMPLES, rng=0)
    ratio = paired_damage_ratio_ci(*arrays, n_resamples=RESAMPLES, rng=0)
    _same(seeded_ratio, ratio)
    _same(seeded_ratio.denominator, ratio.denominator)
    assert seeded_ratio.nonfinite_resamples == ratio.nonfinite_resamples


def test_one_seed_matches_the_existing_two_array_deltas() -> None:
    reference, variant = _row(0), _row(1, accuracy=0.6)
    _same(
        paired_seeded_ci(
            reference, [variant], quantity="delta_accuracy", n_resamples=RESAMPLES, rng=0
        ),
        paired_delta_accuracy_ci(reference[1], variant[1], n_resamples=RESAMPLES, rng=0),
    )
    _same(
        paired_seeded_ci(
            reference, [variant], quantity="delta_confidence", n_resamples=RESAMPLES, rng=0
        ),
        paired_delta_confidence_ci(reference[0], variant[0], n_resamples=RESAMPLES, rng=0),
    )


def test_one_seed_matches_the_existing_residual() -> None:
    ref_fit, ref_report = _scored(0, N_FIT), _scored(1, N_ITEMS)
    var_fit, var_report = _scored(2, N_FIT), _scored(3, N_ITEMS)
    seeded = paired_seeded_residual_ece_ci(
        ref_fit, ref_report, [var_fit], [var_report], n_resamples=RESAMPLES, rng=0
    )
    existing = paired_residual_ece_ci(
        ref_fit, ref_report, var_fit, var_report, n_resamples=RESAMPLES, rng=0
    )
    _same(seeded, existing)
    assert seeded.refused_resamples == existing.refused_resamples


def test_the_variant_ece_averages_the_per_seed_eces() -> None:
    reference = _row(0)
    seeds = [_row(seed, accuracy=0.6) for seed in (1, 2, 3)]
    per_seed = [ece_from_confidence(conf, corr) for conf, corr in seeds]
    pooled = ece_from_confidence(
        np.concatenate([conf for conf, _ in seeds]), np.concatenate([corr for _, corr in seeds])
    )
    expected = float(np.mean(per_seed)) - ece_from_confidence(*reference)
    point = seeded_quantity("delta_ece", reference, seeds)
    assert point == pytest.approx(expected)
    assert point != pytest.approx(pooled - ece_from_confidence(*reference))


@pytest.mark.parametrize(
    "quantity", ["delta_accuracy", "delta_confidence", "delta_ece", "damage_gap"]
)
def test_seed_copies_equal_to_the_reference_give_zero_in_every_resample(
    quantity: SeededQuantity,
) -> None:
    reference = _row(0)
    interval = paired_seeded_ci(
        reference, [reference, reference], quantity=quantity, n_resamples=RESAMPLES, rng=0
    )
    assert (interval.point, interval.low, interval.high) == (0.0, 0.0, 0.0)


def test_refuses_no_seeds_and_misaligned_copies() -> None:
    reference = _row(0)
    with pytest.raises(ValueError, match="at least one seed"):
        paired_seeded_ci(reference, [], quantity="delta_ece", n_resamples=RESAMPLES)
    short = (reference[0][:-1], reference[1][:-1])
    with pytest.raises(ValueError, match="cover the reference's items"):
        paired_seeded_ci(reference, [short], quantity="delta_ece", n_resamples=RESAMPLES)
