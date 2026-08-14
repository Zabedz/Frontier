"""Paired bootstrap confidence intervals via ``scipy.stats.bootstrap``.

Every statistic runs on 1-D per-item arrays, so ``paired=True`` applies one index vector to
confidence, correctness, and both variants of a delta together (methodology section 6). The
intervals are percentile: BCa's acceleration term is undefined on the constant resample
distribution a degenerate fixture (all-correct, identical variants) produces.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import DegenerateDataWarning, bootstrap  # type: ignore[import-untyped]

from frontier.metrics._array import CorrectArray, FloatArray, IntArray, LabelArray, ProbMatrix
from frontier.metrics.binning import BinScheme
from frontier.metrics.calibration import (
    DEFAULT_BINS,
    Weighting,
    ece_from_confidence,
    top_label,
)
from frontier.metrics.recalibration import (
    TemperatureFitError,
    apply_temperature,
    fit_temperature,
)

DEFAULT_RESAMPLES = 9999


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A point estimate on the full sample with its bootstrap interval."""

    point: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies wholly above or wholly below zero."""
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            return False
        return self.low > 0.0 or self.high < 0.0


@dataclass(frozen=True, slots=True)
class RatioInterval:
    """A ratio of two relative changes, with the diagnostics that say whether to trust it.

    A denominator that can approach zero gives the resample distribution a second mode at
    the opposite sign, so ``denominator`` carries its own interval for the caller to check
    the sign, and ``nonfinite_resamples`` counts the draws where the ratio was undefined.
    ``low`` and ``high`` go ``nan`` as soon as one resample is non-finite, with no
    recomputation from the finite ones.
    """

    point: float
    low: float
    high: float
    denominator: ConfidenceInterval
    nonfinite_resamples: int
    n_resamples: int

    @property
    def usable(self) -> bool:
        """Whether the interval can be quoted."""
        return self.nonfinite_resamples == 0 and self.denominator.excludes_zero


def _normalise_rng(rng: np.random.Generator | int | None) -> np.random.Generator | None:
    if isinstance(rng, int):
        return np.random.default_rng(rng)
    return rng


def _mean(sample: CorrectArray, axis: int = -1) -> FloatArray:
    reduced: FloatArray = np.mean(sample, axis=axis)
    return reduced


def accuracy_ci(
    correct: CorrectArray,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | int | None = None,
) -> ConfidenceInterval:
    """Percentile bootstrap interval on the mean accuracy."""
    point = float(np.mean(correct))
    result = bootstrap(
        (correct,),
        _mean,
        vectorized=True,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        rng=_normalise_rng(rng),
    )
    interval = result.confidence_interval
    return ConfidenceInterval(point=point, low=float(interval.low), high=float(interval.high))


def ece_ci(
    confidence: FloatArray,
    correct: CorrectArray,
    *,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
    weighting: Weighting = "mass",
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | int | None = None,
) -> ConfidenceInterval:
    """Percentile bootstrap interval on a single-variant ECE."""
    point = ece_from_confidence(
        confidence, correct, n_bins=n_bins, scheme=scheme, weighting=weighting
    )

    def statistic(resampled_confidence: FloatArray, resampled_correct: CorrectArray) -> float:
        return ece_from_confidence(
            resampled_confidence,
            resampled_correct,
            n_bins=n_bins,
            scheme=scheme,
            weighting=weighting,
        )

    result = bootstrap(
        (confidence, correct),
        statistic,
        paired=True,
        vectorized=False,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        rng=_normalise_rng(rng),
    )
    interval = result.confidence_interval
    return ConfidenceInterval(point=point, low=float(interval.low), high=float(interval.high))


def paired_delta_accuracy_ci(
    correct_a: CorrectArray,
    correct_b: CorrectArray,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | int | None = None,
) -> ConfidenceInterval:
    """Percentile bootstrap interval on the paired accuracy delta ``b - a``."""
    point = float(np.mean(correct_b) - np.mean(correct_a))

    def statistic(resampled_a: CorrectArray, resampled_b: CorrectArray) -> float:
        return float(np.mean(resampled_b) - np.mean(resampled_a))

    result = bootstrap(
        (correct_a, correct_b),
        statistic,
        paired=True,
        vectorized=False,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        rng=_normalise_rng(rng),
    )
    interval = result.confidence_interval
    return ConfidenceInterval(point=point, low=float(interval.low), high=float(interval.high))


def paired_delta_ece_ci(
    confidence_a: FloatArray,
    correct_a: CorrectArray,
    confidence_b: FloatArray,
    correct_b: CorrectArray,
    *,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
    weighting: Weighting = "mass",
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | int | None = None,
) -> ConfidenceInterval:
    """Percentile bootstrap interval on the paired ECE delta ``b - a``.

    One index vector resamples all four arrays, so the caller has to pass them item-aligned
    and scored against the same gold.
    """

    def delta(
        conf_a: FloatArray, corr_a: CorrectArray, conf_b: FloatArray, corr_b: CorrectArray
    ) -> float:
        left = ece_from_confidence(
            conf_a, corr_a, n_bins=n_bins, scheme=scheme, weighting=weighting
        )
        right = ece_from_confidence(
            conf_b, corr_b, n_bins=n_bins, scheme=scheme, weighting=weighting
        )
        return right - left

    point = delta(confidence_a, correct_a, confidence_b, correct_b)
    result = bootstrap(
        (confidence_a, correct_a, confidence_b, correct_b),
        delta,
        paired=True,
        vectorized=False,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        rng=_normalise_rng(rng),
    )
    interval = result.confidence_interval
    return ConfidenceInterval(point=point, low=float(interval.low), high=float(interval.high))


def _paired_percentile_ci(
    arrays: tuple[FloatArray | CorrectArray, ...],
    statistic: Callable[..., float],
    *,
    confidence_level: float,
    n_resamples: int,
    rng: np.random.Generator | int | None,
) -> tuple[float, float, FloatArray]:
    """One paired percentile bootstrap; the distribution lets a caller count undefined draws."""
    result = bootstrap(
        arrays,
        statistic,
        paired=True,
        vectorized=False,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        rng=_normalise_rng(rng),
    )
    interval = result.confidence_interval
    distribution: FloatArray = np.asarray(result.bootstrap_distribution, dtype=np.float64)
    return float(interval.low), float(interval.high), distribution


def relative_damages(
    confidence_a: FloatArray,
    correct_a: CorrectArray,
    confidence_b: FloatArray,
    correct_b: CorrectArray,
    *,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
    weighting: Weighting = "mass",
) -> tuple[float, float]:
    """Relative calibration damage and relative accuracy damage of ``b`` against ``a``.

    Both are signed so that positive means ``b`` is the worse model: calibration damage is
    the fractional rise in ECE, accuracy damage the fractional fall in accuracy, which puts
    the two on one scale. Either is ``nan`` when its reference value is zero, marking the
    resample undefined.
    """
    ece_a = ece_from_confidence(
        confidence_a, correct_a, n_bins=n_bins, scheme=scheme, weighting=weighting
    )
    ece_b = ece_from_confidence(
        confidence_b, correct_b, n_bins=n_bins, scheme=scheme, weighting=weighting
    )
    accuracy_a = float(np.mean(correct_a))
    accuracy_b = float(np.mean(correct_b))
    calibration_damage = math.nan if ece_a == 0.0 else (ece_b - ece_a) / ece_a
    accuracy_damage = math.nan if accuracy_a == 0.0 else (accuracy_a - accuracy_b) / accuracy_a
    return calibration_damage, accuracy_damage


def paired_damage_gap_ci(
    confidence_a: FloatArray,
    correct_a: CorrectArray,
    confidence_b: FloatArray,
    correct_b: CorrectArray,
    *,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
    weighting: Weighting = "mass",
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | int | None = None,
) -> ConfidenceInterval:
    """Percentile bootstrap interval on the damage gap, calibration minus accuracy.

    An interval wholly above zero is the direction claim, that compression costs more
    calibration than accuracy in relative terms. Being a difference of two fractions, the
    gap stays estimable where the accuracy damage is near zero and the ratio is not. Both
    damages are recomputed inside every resample, reference values included.
    """

    def gap(
        conf_a: FloatArray, corr_a: CorrectArray, conf_b: FloatArray, corr_b: CorrectArray
    ) -> float:
        calibration_damage, accuracy_damage = relative_damages(
            conf_a, corr_a, conf_b, corr_b, n_bins=n_bins, scheme=scheme, weighting=weighting
        )
        return calibration_damage - accuracy_damage

    point = gap(confidence_a, correct_a, confidence_b, correct_b)
    low, high, _distribution = _paired_percentile_ci(
        (confidence_a, correct_a, confidence_b, correct_b),
        gap,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        rng=rng,
    )
    return ConfidenceInterval(point=point, low=low, high=high)


def paired_damage_ratio_ci(
    confidence_a: FloatArray,
    correct_a: CorrectArray,
    confidence_b: FloatArray,
    correct_b: CorrectArray,
    *,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
    weighting: Weighting = "mass",
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | int | None = None,
) -> RatioInterval:
    """Percentile bootstrap on the damage ratio, calibration loss over accuracy loss.

    The ratio is the magnitude behind "ECE degrades twice as fast". Its denominator is
    bootstrapped alongside it, because a ratio whose denominator interval spans zero has no
    meaningful quantiles however tight they look. Read ``RatioInterval.usable`` before
    quoting the interval.
    """

    def ratio(
        conf_a: FloatArray, corr_a: CorrectArray, conf_b: FloatArray, corr_b: CorrectArray
    ) -> float:
        calibration_damage, accuracy_damage = relative_damages(
            conf_a, corr_a, conf_b, corr_b, n_bins=n_bins, scheme=scheme, weighting=weighting
        )
        if accuracy_damage == 0.0:
            return math.nan
        return calibration_damage / accuracy_damage

    def denominator(
        conf_a: FloatArray, corr_a: CorrectArray, conf_b: FloatArray, corr_b: CorrectArray
    ) -> float:
        return relative_damages(
            conf_a, corr_a, conf_b, corr_b, n_bins=n_bins, scheme=scheme, weighting=weighting
        )[1]

    arrays = (confidence_a, correct_a, confidence_b, correct_b)
    point = ratio(confidence_a, correct_a, confidence_b, correct_b)
    # A nan quantile from an undefined resample draws a BCa warning that does not apply here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegenerateDataWarning)
        low, high, distribution = _paired_percentile_ci(
            arrays, ratio, confidence_level=confidence_level, n_resamples=n_resamples, rng=rng
        )
    nonfinite = int(np.count_nonzero(~np.isfinite(distribution)))
    denominator_low, denominator_high, _denominator_distribution = _paired_percentile_ci(
        arrays, denominator, confidence_level=confidence_level, n_resamples=n_resamples, rng=rng
    )
    return RatioInterval(
        point=point,
        low=math.nan if nonfinite else low,
        high=math.nan if nonfinite else high,
        denominator=ConfidenceInterval(
            point=denominator(*arrays), low=denominator_low, high=denominator_high
        ),
        nonfinite_resamples=nonfinite,
        n_resamples=int(distribution.size),
    )


@dataclass(frozen=True, slots=True)
class ScoredItems:
    """One half of one variant's held-out split, as the recalibration statistic needs it."""

    probs: ProbMatrix
    gold: LabelArray
    n_options: IntArray


@dataclass(frozen=True, slots=True)
class ResidualInterval:
    """A difference of post-recalibration ECEs, with the resamples that refused a fit."""

    point: float
    low: float
    high: float
    refused_resamples: int
    n_resamples: int

    @property
    def excludes_zero(self) -> bool:
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            return False
        return self.low > 0.0 or self.high < 0.0

    @property
    def usable(self) -> bool:
        """Whether every resample produced a temperature."""
        return self.refused_resamples == 0


def _residual_ece(
    fit: ScoredItems,
    report: ScoredItems,
    fit_rows: IntArray,
    report_rows: IntArray,
    *,
    n_bins: int,
) -> float:
    """ECE on the report rows after a temperature fitted on the fit rows."""
    temperature = fit_temperature(fit.probs[fit_rows], fit.gold[fit_rows], fit.n_options[fit_rows])
    scaled = apply_temperature(
        report.probs[report_rows], temperature, report.n_options[report_rows]
    )
    confidence, correct = top_label(scaled, report.gold[report_rows])
    return ece_from_confidence(confidence, correct, n_bins=n_bins)


def paired_residual_ece_ci(
    reference_fit: ScoredItems,
    reference_report: ScoredItems,
    variant_fit: ScoredItems,
    variant_report: ScoredItems,
    *,
    n_bins: int = DEFAULT_BINS,
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | int | None = None,
) -> ResidualInterval:
    """Interval on ``residual(variant) - residual(reference)`` after recalibrating each.

    Resamples the fit half and the report half separately, because they hold different
    items, and shares each index vector across the two variants so the difference stays
    paired. The temperature is refitted inside every resample: holding it at its full
    sample value would report the residual as if the fit were free of error, and the fit
    noise is what puts a perfectly calibrated model's removed fraction near -13%.

    ``scipy.stats.bootstrap`` takes one index vector across every array it is given, so it
    cannot express a statistic drawing from two samples of different length.

    A resample whose fit refuses (see ``TemperatureFitError``) is counted and dropped; the
    interval comes from the resamples that produced a temperature, and ``usable`` is False
    whenever any refused.
    """
    if reference_report.gold.shape[0] != variant_report.gold.shape[0]:
        raise ValueError(
            f"report halves differ in length: reference {reference_report.gold.shape[0]}, "
            f"variant {variant_report.gold.shape[0]}"
        )
    if reference_fit.gold.shape[0] != variant_fit.gold.shape[0]:
        raise ValueError(
            f"fit halves differ in length: reference {reference_fit.gold.shape[0]}, "
            f"variant {variant_fit.gold.shape[0]}"
        )
    generator = _normalise_rng(rng) or np.random.default_rng()
    n_fit = reference_fit.gold.shape[0]
    n_report = reference_report.gold.shape[0]
    point = _residual_ece(
        variant_fit, variant_report, np.arange(n_fit), np.arange(n_report), n_bins=n_bins
    ) - _residual_ece(
        reference_fit, reference_report, np.arange(n_fit), np.arange(n_report), n_bins=n_bins
    )
    drawn: list[float] = []
    refused = 0
    for _ in range(n_resamples):
        fit_rows = generator.integers(0, n_fit, size=n_fit)
        report_rows = generator.integers(0, n_report, size=n_report)
        try:
            drawn.append(
                _residual_ece(variant_fit, variant_report, fit_rows, report_rows, n_bins=n_bins)
                - _residual_ece(
                    reference_fit, reference_report, fit_rows, report_rows, n_bins=n_bins
                )
            )
        except TemperatureFitError:
            refused += 1
    if not drawn:  # defensive: the full-sample fit above refuses before this can happen
        raise TemperatureFitError(f"every one of {n_resamples} resamples refused a temperature fit")
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(np.asarray(drawn), [alpha, 1.0 - alpha])
    return ResidualInterval(
        point=point,
        low=float(low),
        high=float(high),
        refused_resamples=refused,
        n_resamples=n_resamples,
    )
