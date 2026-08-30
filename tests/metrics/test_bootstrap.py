"""Bootstrap intervals, the paired-resampling guard, and the damage gap and ratio.

A paired resample indexes both arrays with one index vector; the identical-variant tests
are the regression against independent resampling.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import numpy.typing as npt
import pytest
from conftest import make_calibrated_confidence, make_calibrated_softmax, make_gold, make_softmax
from scipy.stats import DegenerateDataWarning, bootstrap  # type: ignore[import-untyped]

from frontier.metrics.bootstrap import (
    ScoredItems,
    accuracy_ci,
    ece_ci,
    paired_damage_gap_ci,
    paired_damage_ratio_ci,
    paired_delta_accuracy_ci,
    paired_delta_ece_ci,
    paired_residual_ece_ci,
    relative_damages,
    residual_ece,
)
from frontier.metrics.calibration import ece_from_confidence, top_label

RESAMPLES = 999
RATE = 0.7
N_FIT = 30
N_REPORT = 60
N_OPTIONS = 4
SHARPEN = 0.5
POISON_LOGIT = 60.0  # a confidently wrong item, so a draw holding it runs past the bound


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


def test_relative_damages_signs_positive_when_the_second_variant_is_worse() -> None:
    confidence_a, correct_a = make_calibrated_confidence(2000, np.random.default_rng(20))
    confidence_b = np.clip(confidence_a + 0.15, 0.0, 1.0)
    correct_b = correct_a.copy()
    correct_b[:100] = False
    calibration_damage, accuracy_damage = relative_damages(
        confidence_a, correct_a, confidence_b, correct_b, n_bins=10
    )
    assert calibration_damage > 0.0
    assert accuracy_damage > 0.0


def test_relative_damages_are_nan_when_a_reference_is_zero() -> None:
    perfect = np.array([1.0, 1.0, 1.0, 1.0])
    hit = np.array([True, True, True, True])
    _calibration_damage, accuracy_damage = relative_damages(
        perfect, np.zeros(4, dtype=np.bool_), perfect, hit, n_bins=2
    )
    assert math.isnan(accuracy_damage)


def test_damage_gap_ci_is_zero_on_identical_variants() -> None:
    confidence, correct = make_calibrated_confidence(1500, np.random.default_rng(21))
    interval = paired_damage_gap_ci(
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
    assert not interval.excludes_zero


def test_damage_gap_point_matches_the_hand_computed_difference() -> None:
    confidence_a, correct_a = make_calibrated_confidence(2000, np.random.default_rng(22))
    confidence_b = np.clip(confidence_a + 0.1, 0.0, 1.0)
    correct_b = correct_a.copy()
    correct_b[:80] = False
    calibration_damage, accuracy_damage = relative_damages(
        confidence_a, correct_a, confidence_b, correct_b, n_bins=10
    )
    interval = paired_damage_gap_ci(
        confidence_a, correct_a, confidence_b, correct_b, n_bins=10, n_resamples=RESAMPLES, rng=7
    )
    assert interval.point == pytest.approx(calibration_damage - accuracy_damage)


def test_damage_gap_ci_excludes_zero_when_only_calibration_moves() -> None:
    confidence_a, correct_a = make_calibrated_confidence(3000, np.random.default_rng(23))
    confidence_b = np.clip(confidence_a + 0.2, 0.0, 1.0)
    interval = paired_damage_gap_ci(
        confidence_a,
        correct_a,
        confidence_b,
        correct_a.copy(),
        n_bins=10,
        n_resamples=RESAMPLES,
        rng=7,
    )
    assert interval.point > 0.0
    assert interval.excludes_zero


def test_damage_ratio_point_matches_the_hand_computed_quotient() -> None:
    confidence_a, correct_a = make_calibrated_confidence(2000, np.random.default_rng(24))
    confidence_b = np.clip(confidence_a + 0.1, 0.0, 1.0)
    correct_b = correct_a.copy()
    correct_b[:120] = False
    calibration_damage, accuracy_damage = relative_damages(
        confidence_a, correct_a, confidence_b, correct_b, n_bins=10
    )
    ratio = paired_damage_ratio_ci(
        confidence_a, correct_a, confidence_b, correct_b, n_bins=10, n_resamples=RESAMPLES, rng=7
    )
    assert ratio.point == pytest.approx(calibration_damage / accuracy_damage)
    assert ratio.n_resamples == RESAMPLES


def test_damage_ratio_is_unusable_when_the_accuracy_damage_is_zero() -> None:
    confidence_a, correct_a = make_calibrated_confidence(2000, np.random.default_rng(25))
    confidence_b = np.clip(confidence_a + 0.2, 0.0, 1.0)
    ratio = paired_damage_ratio_ci(
        confidence_a,
        correct_a,
        confidence_b,
        correct_a.copy(),
        n_bins=10,
        n_resamples=RESAMPLES,
        rng=7,
    )
    assert math.isnan(ratio.point)
    assert math.isnan(ratio.low) and math.isnan(ratio.high)
    assert ratio.nonfinite_resamples == RESAMPLES
    assert not ratio.usable


def test_damage_ratio_is_unusable_when_the_denominator_interval_spans_zero() -> None:
    confidence_a, correct_a = make_calibrated_confidence(2000, np.random.default_rng(26))
    confidence_b = np.clip(confidence_a + 0.2, 0.0, 1.0)
    correct_b = correct_a.copy()
    correct_b[:3] = ~correct_b[:3]
    ratio = paired_damage_ratio_ci(
        confidence_a, correct_a, confidence_b, correct_b, n_bins=10, n_resamples=RESAMPLES, rng=7
    )
    assert not ratio.denominator.excludes_zero
    assert not ratio.usable


def test_damage_ratio_emits_no_degenerate_data_warning() -> None:
    confidence_a, correct_a = make_calibrated_confidence(500, np.random.default_rng(27))
    confidence_b = np.clip(confidence_a + 0.2, 0.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DegenerateDataWarning)
        paired_damage_ratio_ci(
            confidence_a,
            correct_a,
            confidence_b,
            correct_a.copy(),
            n_bins=10,
            n_resamples=RESAMPLES,
            rng=7,
        )


def test_relative_damages_match_a_closed_form_fixture() -> None:
    """One occupied bin makes ECE exactly ``|accuracy - confidence|``."""
    reference_confidence = np.full(100, 0.90)
    reference_correct = np.array([True] * 80 + [False] * 20)
    variant_confidence = np.full(100, 0.91)
    variant_correct = np.array([True] * 76 + [False] * 24)
    calibration_damage, accuracy_damage = relative_damages(
        reference_confidence,
        reference_correct,
        variant_confidence,
        variant_correct,
        n_bins=10,
    )
    assert calibration_damage == pytest.approx(0.50)  # (0.15 - 0.10) / 0.10
    assert accuracy_damage == pytest.approx(0.05)  # (0.80 - 0.76) / 0.80


def test_damage_gap_and_ratio_match_the_closed_form_fixture() -> None:
    reference_confidence = np.full(100, 0.90)
    reference_correct = np.array([True] * 80 + [False] * 20)
    variant_confidence = np.full(100, 0.91)
    variant_correct = np.array([True] * 76 + [False] * 24)
    arrays = (reference_confidence, reference_correct, variant_confidence, variant_correct)
    gap = paired_damage_gap_ci(*arrays, n_bins=10, n_resamples=RESAMPLES, rng=7)
    ratio = paired_damage_ratio_ci(*arrays, n_bins=10, n_resamples=RESAMPLES, rng=7)
    assert gap.point == pytest.approx(0.45)  # 0.50 - 0.05
    assert ratio.point == pytest.approx(10.0)  # 0.50 / 0.05


def test_damage_ratio_is_usable_when_the_damages_are_both_clearly_signed() -> None:
    confidence_a, correct_a = make_calibrated_confidence(2000, np.random.default_rng(24))
    confidence_b = np.clip(confidence_a + 0.1, 0.0, 1.0)
    correct_b = correct_a.copy()
    correct_b[:120] = False
    ratio = paired_damage_ratio_ci(
        confidence_a, correct_a, confidence_b, correct_b, n_bins=10, n_resamples=RESAMPLES, rng=7
    )
    assert ratio.usable
    assert ratio.nonfinite_resamples == 0
    assert ratio.denominator.excludes_zero
    assert math.isfinite(ratio.low) and math.isfinite(ratio.high)


def test_damage_ratio_denominator_clause_alone_makes_it_unusable() -> None:
    """Every resample is defined here, so only the unsigned denominator withholds the ratio."""
    confidence_a, correct_a = make_calibrated_confidence(4000, np.random.default_rng(31))
    confidence_b = np.clip(confidence_a + 0.2, 0.0, 1.0)
    correct_b = correct_a.copy()
    hits = np.flatnonzero(correct_a)
    misses = np.flatnonzero(~correct_a)
    correct_b[hits[:500]] = False
    correct_b[misses[:480]] = True
    ratio = paired_damage_ratio_ci(
        confidence_a, correct_a, confidence_b, correct_b, n_bins=10, n_resamples=199, rng=7
    )
    assert ratio.nonfinite_resamples == 0
    assert not ratio.denominator.excludes_zero
    assert not ratio.usable


def _softmax(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    exponentiated = np.exp(scores - scores.max(axis=1, keepdims=True))
    normalised: npt.NDArray[np.float64] = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    return normalised


def _halves(*, sharpen: float, poison: bool) -> tuple[ScoredItems, ScoredItems]:
    """One variant's fit and report halves, optionally carrying one unfittable item."""
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 2.0, size=(N_FIT + N_REPORT, N_OPTIONS))
    gold = np.asarray([rng.choice(N_OPTIONS, p=row) for row in _softmax(logits)], dtype=np.intp)
    scores = logits / sharpen
    if poison:
        scores[0] = 0.0
        scores[0, (gold[0] + 1) % N_OPTIONS] = POISON_LOGIT
    probs = _softmax(scores)
    n_options = np.full(N_FIT + N_REPORT, N_OPTIONS, dtype=np.intp)
    return (
        ScoredItems(probs=probs[:N_FIT], gold=gold[:N_FIT], n_options=n_options[:N_FIT]),
        ScoredItems(probs=probs[N_FIT:], gold=gold[N_FIT:], n_options=n_options[N_FIT:]),
    )


def test_residual_interval_is_centred_on_the_two_whole_half_residuals() -> None:
    """The point estimate is the difference the pair table stores, computed one way."""
    reference_fit, reference_report = _halves(sharpen=1.0, poison=False)
    variant_fit, variant_report = _halves(sharpen=SHARPEN, poison=False)
    interval = paired_residual_ece_ci(
        reference_fit, reference_report, variant_fit, variant_report, n_resamples=199, rng=0
    )
    variant = residual_ece(variant_fit, variant_report)
    reference = residual_ece(reference_fit, reference_report)
    assert interval.point == variant - reference
    assert interval.refused_resamples == 0
    assert interval.usable


def test_an_explicit_full_index_matches_the_default() -> None:
    fit, report = _halves(sharpen=SHARPEN, poison=False)
    whole = residual_ece(fit, report)
    spelled_out = residual_ece(
        fit, report, np.arange(fit.gold.shape[0]), np.arange(report.gold.shape[0])
    )
    assert whole == spelled_out


def test_resamples_that_refuse_a_fit_are_counted_and_dropped() -> None:
    """One confidently wrong item; a draw holding it more than once runs to the bound.

    The full sample still fits, so the interval is built from the surviving draws and
    ``usable`` is what says the distribution was thinned.
    """
    reference_fit, reference_report = _halves(sharpen=1.0, poison=False)
    variant_fit, variant_report = _halves(sharpen=SHARPEN, poison=True)
    interval = paired_residual_ece_ci(
        reference_fit, reference_report, variant_fit, variant_report, n_resamples=199, rng=0
    )
    assert 0 < interval.refused_resamples < interval.n_resamples
    assert not interval.usable
    assert math.isfinite(interval.low) and math.isfinite(interval.high)
    assert interval.low <= interval.point <= interval.high


def test_report_halves_of_different_length_are_refused() -> None:
    fit, report = _halves(sharpen=SHARPEN, poison=False)
    short = ScoredItems(
        probs=report.probs[:-1], gold=report.gold[:-1], n_options=report.n_options[:-1]
    )
    with pytest.raises(ValueError, match="report halves differ in length"):
        paired_residual_ece_ci(fit, short, fit, report, n_resamples=19, rng=0)


def test_fit_halves_of_different_length_are_refused() -> None:
    fit, report = _halves(sharpen=SHARPEN, poison=False)
    short = ScoredItems(probs=fit.probs[:-1], gold=fit.gold[:-1], n_options=fit.n_options[:-1])
    with pytest.raises(ValueError, match="fit halves differ in length"):
        paired_residual_ece_ci(short, report, fit, report, n_resamples=19, rng=0)
