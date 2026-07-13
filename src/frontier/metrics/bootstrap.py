"""Paired bootstrap confidence intervals via ``scipy.stats.bootstrap``.

Every statistic runs on 1-D per-item arrays, so ``paired=True`` resamples one index
vector and applies it to confidence and correctness together; the two variants of a
delta never resample independently (methodology section 6). Percentile intervals,
not BCa: the degenerate fixtures (all-correct, identical variants) give a constant
resample distribution on which BCa's acceleration term is undefined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import bootstrap  # type: ignore[import-untyped]

from frontier.metrics._array import CorrectArray, FloatArray
from frontier.metrics.binning import BinScheme
from frontier.metrics.calibration import DEFAULT_BINS, Weighting, ece_from_confidence

DEFAULT_RESAMPLES = 9999


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A point estimate on the full sample with its bootstrap interval."""

    point: float
    low: float
    high: float


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
    """Percentile bootstrap interval on a single-variant ECE.

    ``confidence`` and ``correct`` resample together under ``paired=True``.
    """
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
    """Percentile bootstrap interval on the paired accuracy delta ``b - a``.

    One index vector resamples ``correct_a`` and ``correct_b`` together, so the
    delta CI reflects the paired difference, never independent resampling.
    """
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

    A single index vector resamples all four arrays together, so ``confidence_a`` /
    ``correct_a`` and ``confidence_b`` / ``correct_b`` stay aligned across every
    resample. The gold labels are shared across variants, so ``correct_a`` and
    ``correct_b`` are computed against the same gold before entering the bootstrap.
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
