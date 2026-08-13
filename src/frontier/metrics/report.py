"""Assemble the calibration battery and map it onto the frozen ``schema.Quality`` contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frontier.metrics._array import LabelArray, ProbMatrix, check_predictions
from frontier.metrics.bootstrap import ConfidenceInterval
from frontier.metrics.calibration import (
    DEFAULT_BINS,
    DEFAULT_SWEEP,
    ReliabilityCurve,
    ece_from_confidence,
    reliability_from_confidence,
    top_label,
)
from frontier.metrics.scoring import brier_decomposition, brier_score, nll
from frontier.schema import Quality


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """The full per-variant calibration battery.

    ``reliability`` is carried for the plotting step and stored alongside the row; it has no
    ``Quality`` field.
    """

    accuracy: float
    ece_equal_width: float
    ece_equal_mass_ace: float
    ece_bin_sweep: dict[int, float]
    brier: float
    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float
    nll: float
    reliability: ReliabilityCurve


def calibration_report(
    probs: ProbMatrix,
    gold: LabelArray,
    *,
    n_bins: int = DEFAULT_BINS,
    sweep: tuple[int, ...] = DEFAULT_SWEEP,
) -> CalibrationReport:
    """Compute the calibration battery, sharing one ``top_label`` reduction across it."""
    check_predictions(probs, gold)
    confidence, correct = top_label(probs, gold)
    decomposition = brier_decomposition(probs, gold, n_bins=n_bins)
    return CalibrationReport(
        accuracy=float(np.mean(correct)),
        ece_equal_width=ece_from_confidence(
            confidence, correct, n_bins=n_bins, scheme="equal_width", weighting="mass"
        ),
        ece_equal_mass_ace=ece_from_confidence(
            confidence, correct, n_bins=n_bins, scheme="equal_mass", weighting="mass"
        ),
        ece_bin_sweep={
            n: ece_from_confidence(confidence, correct, n_bins=n, scheme="equal_width")
            for n in sweep
        },
        brier=brier_score(probs, gold),
        brier_reliability=decomposition.reliability,
        brier_resolution=decomposition.resolution,
        brier_uncertainty=decomposition.uncertainty,
        nll=nll(probs, gold),
        reliability=reliability_from_confidence(confidence, correct, n_bins, "equal_width"),
    )


def to_quality(
    report: CalibrationReport,
    *,
    accuracy_ci: ConfidenceInterval,
    ece_ci: ConfidenceInterval,
    perplexity: float,
    temperature: float = 1.0,
    temperature_scaled: bool = False,
) -> Quality:
    """Map a report and its two bootstrap intervals onto ``schema.Quality``.

    ``ece_ci`` has to be the interval on the equal-width ECE at the report's bin count, to
    match the ``ece_equal_width`` it lands beside; the pairing is unchecked.
    """
    return Quality(
        accuracy=report.accuracy,
        accuracy_ci_low=accuracy_ci.low,
        accuracy_ci_high=accuracy_ci.high,
        ece_equal_width=report.ece_equal_width,
        ece_equal_mass_ace=report.ece_equal_mass_ace,
        ece_bin_sweep=report.ece_bin_sweep,
        ece_ci_low=ece_ci.low,
        ece_ci_high=ece_ci.high,
        brier=report.brier,
        brier_reliability=report.brier_reliability,
        brier_resolution=report.brier_resolution,
        brier_uncertainty=report.brier_uncertainty,
        nll=report.nll,
        perplexity=perplexity,
        temperature_scaled=temperature_scaled,
        temperature=temperature,
    )
