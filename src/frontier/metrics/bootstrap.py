"""Paired bootstrap confidence intervals over the per-item arrays.

The delta statistics run on 1-D per-item arrays through ``scipy.stats.bootstrap``, so
``paired=True`` applies one index vector to confidence, correctness, and both variants of a
delta together (methodology section 6). The residual statistic draws from two halves of
differing length and is hand-rolled at the foot of the file. The intervals are percentile:
BCa's acceleration term is undefined on the constant resample distribution a degenerate
fixture (all-correct, identical variants) produces.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

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
# before_1, after_1, before_2, after_2 for a shift difference.
SHIFT_ROWS = 4

ShiftMetric = Literal["ece", "confidence"]
SeededQuantity = Literal[
    "delta_accuracy",
    "delta_confidence",
    "delta_ece",
    "damage_gap",
    "accuracy_damage",
    "damage_ratio",
]


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


def paired_delta_confidence_ci(
    confidence_a: FloatArray,
    confidence_b: FloatArray,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | int | None = None,
) -> ConfidenceInterval:
    """Percentile bootstrap interval on the paired mean-confidence delta ``b - a``.

    The ECE delta cannot tell a model that grew overconfident from one that simply got more
    answers wrong. This separates them: it moves only when compression changed what the
    model says about itself. See methodology section 6.
    """
    point = float(np.mean(confidence_b) - np.mean(confidence_a))

    def statistic(resampled_a: FloatArray, resampled_b: FloatArray) -> float:
        return float(np.mean(resampled_b) - np.mean(resampled_a))

    result = bootstrap(
        (confidence_a, confidence_b),
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


def paired_shift_difference_ci(
    confidence: Sequence[FloatArray],
    correct: Sequence[CorrectArray],
    *,
    metric: ShiftMetric,
    n_bins: int = DEFAULT_BINS,
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | int | None = None,
) -> ConfidenceInterval:
    """Interval on ``(after_1 - before_1) - (after_2 - before_2)`` for one metric.

    ``confidence`` and ``correct`` hold four rows each, ordered before_1, after_1, before_2,
    after_2, all on the same items, so one index vector resamples all eight arrays.
    ``metric="confidence"`` contrasts mean stated confidence and leaves ``correct`` unread.
    """
    if len(confidence) != SHIFT_ROWS or len(correct) != SHIFT_ROWS:
        raise ValueError(
            f"a shift difference needs {SHIFT_ROWS} rows of each array, got "
            f"{len(confidence)} confidence and {len(correct)} correct"
        )
    lengths = {int(array.shape[0]) for array in (*confidence, *correct)}
    if len(lengths) != 1:
        raise ValueError(f"the four rows must cover the same items, got lengths {sorted(lengths)}")

    def value(conf: FloatArray, corr: CorrectArray) -> float:
        if metric == "ece":
            return ece_from_confidence(conf, corr, n_bins=n_bins)
        return float(np.mean(conf))

    def statistic(*arrays: FloatArray | CorrectArray) -> float:
        values = [
            value(
                np.asarray(arrays[row], dtype=np.float64),
                np.asarray(arrays[SHIFT_ROWS + row], dtype=np.bool_),
            )
            for row in range(SHIFT_ROWS)
        ]
        return (values[1] - values[0]) - (values[3] - values[2])

    arrays: tuple[FloatArray | CorrectArray, ...] = (*confidence, *correct)
    low, high, _distribution = _paired_percentile_ci(
        arrays, statistic, confidence_level=confidence_level, n_resamples=n_resamples, rng=rng
    )
    return ConfidenceInterval(point=statistic(*arrays), low=low, high=high)


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


def residual_ece(
    fit: ScoredItems,
    report: ScoredItems,
    fit_rows: IntArray | None = None,
    report_rows: IntArray | None = None,
    *,
    n_bins: int = DEFAULT_BINS,
) -> float:
    """ECE on the report rows after a temperature fitted on the fit rows.

    ``None`` for either index takes that half whole, which is the point estimate the
    interval below is centred on. One producer, so a table cannot carry a residual and a
    difference of residuals that disagree.
    """
    take_fit = np.arange(fit.gold.shape[0]) if fit_rows is None else fit_rows
    take_report = np.arange(report.gold.shape[0]) if report_rows is None else report_rows
    temperature = fit_temperature(fit.probs[take_fit], fit.gold[take_fit], fit.n_options[take_fit])
    scaled = apply_temperature(
        report.probs[take_report], temperature, report.n_options[take_report]
    )
    confidence, correct = top_label(scaled, report.gold[take_report])
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

    The resampling is hand-rolled. ``scipy.stats.bootstrap`` has nowhere to count the
    resamples whose fit refused.

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
    point = residual_ece(variant_fit, variant_report, n_bins=n_bins) - residual_ece(
        reference_fit, reference_report, n_bins=n_bins
    )
    drawn: list[float] = []
    refused = 0
    for _ in range(n_resamples):
        fit_rows = generator.integers(0, n_fit, size=n_fit)
        report_rows = generator.integers(0, n_report, size=n_report)
        try:
            drawn.append(
                residual_ece(variant_fit, variant_report, fit_rows, report_rows, n_bins=n_bins)
                - residual_ece(
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


def _seed_mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def seeded_quantity(
    quantity: SeededQuantity,
    reference: tuple[FloatArray, CorrectArray],
    seeds: Sequence[tuple[FloatArray, CorrectArray]],
    *,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
    weighting: Weighting = "mass",
) -> float:
    """One paired quantity with the variant side averaged over its seed copies.

    Each seed copy is scored on its own and its accuracy, ECE, and mean confidence are
    averaged, so every ECE carries the same finite-sample floor as the reference's. Pooling
    the copies into one sample would lower the variant's floor and flatter its calibration.
    The damages follow ``relative_damages``: ``nan`` where a reference value is zero.
    """
    ref_confidence, ref_correct = reference
    ece_ref = ece_from_confidence(
        ref_confidence, ref_correct, n_bins=n_bins, scheme=scheme, weighting=weighting
    )
    accuracy_ref = float(np.mean(ref_correct))
    ece_var = _seed_mean(
        [
            ece_from_confidence(conf, corr, n_bins=n_bins, scheme=scheme, weighting=weighting)
            for conf, corr in seeds
        ]
    )
    accuracy_var = _seed_mean([float(np.mean(corr)) for _conf, corr in seeds])
    if quantity == "delta_accuracy":
        return accuracy_var - accuracy_ref
    if quantity == "delta_confidence":
        return _seed_mean([float(np.mean(conf)) for conf, _corr in seeds]) - float(
            np.mean(ref_confidence)
        )
    if quantity == "delta_ece":
        return ece_var - ece_ref
    calibration_damage = math.nan if ece_ref == 0.0 else (ece_var - ece_ref) / ece_ref
    accuracy_damage = (
        math.nan if accuracy_ref == 0.0 else (accuracy_ref - accuracy_var) / accuracy_ref
    )
    if quantity == "accuracy_damage":
        return accuracy_damage
    if quantity == "damage_gap":
        return calibration_damage - accuracy_damage
    return math.nan if accuracy_damage == 0.0 else calibration_damage / accuracy_damage


def _seeded_arrays(
    reference: tuple[FloatArray, CorrectArray], seeds: Sequence[tuple[FloatArray, CorrectArray]]
) -> tuple[FloatArray | CorrectArray, ...]:
    if not seeds:
        raise ValueError("a seeded statistic needs at least one seed copy")
    lengths = {int(array.shape[0]) for pair in (reference, *seeds) for array in pair}
    if len(lengths) != 1:
        raise ValueError(f"every seed copy must cover the reference's items, got {sorted(lengths)}")
    return tuple(array for pair in (reference, *seeds) for array in pair)


def _seeded_statistic(
    quantity: SeededQuantity, *, n_bins: int, scheme: BinScheme, weighting: Weighting
) -> Callable[..., float]:
    def statistic(*arrays: FloatArray | CorrectArray) -> float:
        pairs = [
            (
                np.asarray(arrays[index], dtype=np.float64),
                np.asarray(arrays[index + 1], dtype=np.bool_),
            )
            for index in range(0, len(arrays), 2)
        ]
        return seeded_quantity(
            quantity, pairs[0], pairs[1:], n_bins=n_bins, scheme=scheme, weighting=weighting
        )

    return statistic


def paired_seeded_ci(
    reference: tuple[FloatArray, CorrectArray],
    seeds: Sequence[tuple[FloatArray, CorrectArray]],
    *,
    quantity: SeededQuantity,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
    weighting: Weighting = "mass",
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | int | None = None,
) -> ConfidenceInterval:
    """Paired percentile interval on ``quantity``, the variant averaged over its seeds.

    ``reference`` and every seed copy are ``(confidence, correct)`` on the same items in the
    same order, so one index vector resamples them all.
    """
    arrays = _seeded_arrays(reference, seeds)
    statistic = _seeded_statistic(quantity, n_bins=n_bins, scheme=scheme, weighting=weighting)
    low, high, _distribution = _paired_percentile_ci(
        arrays, statistic, confidence_level=confidence_level, n_resamples=n_resamples, rng=rng
    )
    return ConfidenceInterval(point=statistic(*arrays), low=low, high=high)


def paired_seeded_ratio_ci(
    reference: tuple[FloatArray, CorrectArray],
    seeds: Sequence[tuple[FloatArray, CorrectArray]],
    *,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
    weighting: Weighting = "mass",
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | int | None = None,
) -> RatioInterval:
    """The damage ratio over seed copies, with ``paired_damage_ratio_ci``'s usability rule."""
    arrays = _seeded_arrays(reference, seeds)
    ratio = _seeded_statistic("damage_ratio", n_bins=n_bins, scheme=scheme, weighting=weighting)
    denominator = _seeded_statistic(
        "accuracy_damage", n_bins=n_bins, scheme=scheme, weighting=weighting
    )
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
        point=ratio(*arrays),
        low=math.nan if nonfinite else low,
        high=math.nan if nonfinite else high,
        denominator=ConfidenceInterval(
            point=denominator(*arrays), low=denominator_low, high=denominator_high
        ),
        nonfinite_resamples=nonfinite,
        n_resamples=int(distribution.size),
    )


def paired_seeded_residual_ece_ci(
    reference_fit: ScoredItems,
    reference_report: ScoredItems,
    variant_fits: Sequence[ScoredItems],
    variant_reports: Sequence[ScoredItems],
    *,
    n_bins: int = DEFAULT_BINS,
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | int | None = None,
) -> ResidualInterval:
    """``paired_residual_ece_ci`` with the variant's residual averaged over seed copies.

    Each copy fits its own temperature on the shared fit rows and reports on the shared
    report rows. A resample is refused when any copy's fit refuses.
    """
    if not variant_fits or len(variant_fits) != len(variant_reports):
        raise ValueError(
            f"need one fit and one report half per seed, got {len(variant_fits)} and "
            f"{len(variant_reports)}"
        )
    n_fit = reference_fit.gold.shape[0]
    n_report = reference_report.gold.shape[0]
    for fit, report in zip(variant_fits, variant_reports, strict=True):
        if fit.gold.shape[0] != n_fit or report.gold.shape[0] != n_report:
            raise ValueError(
                f"every seed copy must match the reference halves ({n_fit} fit, {n_report} "
                f"report), got {fit.gold.shape[0]} and {report.gold.shape[0]}"
            )

    def difference(fit_rows: IntArray | None, report_rows: IntArray | None) -> float:
        variant = _seed_mean(
            [
                residual_ece(fit, report, fit_rows, report_rows, n_bins=n_bins)
                for fit, report in zip(variant_fits, variant_reports, strict=True)
            ]
        )
        return variant - residual_ece(
            reference_fit, reference_report, fit_rows, report_rows, n_bins=n_bins
        )

    generator = _normalise_rng(rng) or np.random.default_rng()
    point = difference(None, None)
    drawn: list[float] = []
    refused = 0
    for _ in range(n_resamples):
        fit_rows = generator.integers(0, n_fit, size=n_fit)
        report_rows = generator.integers(0, n_report, size=n_report)
        try:
            drawn.append(difference(fit_rows, report_rows))
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
