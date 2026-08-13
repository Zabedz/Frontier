"""Calibration metrics, perplexity, and paired bootstrap intervals. CPU-only, and
cross-checked against torchmetrics, netcal, and sklearn oracles.
"""

from __future__ import annotations

from frontier.metrics._array import check_confidence, check_predictions
from frontier.metrics.binning import BinScheme
from frontier.metrics.bootstrap import (
    DEFAULT_RESAMPLES,
    ConfidenceInterval,
    RatioInterval,
    accuracy_ci,
    ece_ci,
    paired_damage_gap_ci,
    paired_damage_ratio_ci,
    paired_delta_accuracy_ci,
    paired_delta_ece_ci,
    relative_damages,
)
from frontier.metrics.calibration import (
    DEFAULT_BINS,
    DEFAULT_SWEEP,
    ReliabilityCurve,
    Weighting,
    ece,
    ece_equal_mass,
    ece_equal_width,
    ece_from_confidence,
    ece_sweep,
    reliability_curve,
    reliability_from_confidence,
    top_label,
)
from frontier.metrics.report import CalibrationReport, calibration_report, to_quality
from frontier.metrics.scoring import BrierDecomposition, brier_decomposition, brier_score, nll

__all__ = [
    "DEFAULT_BINS",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SWEEP",
    "BinScheme",
    "BrierDecomposition",
    "CalibrationReport",
    "ConfidenceInterval",
    "RatioInterval",
    "ReliabilityCurve",
    "Weighting",
    "accuracy_ci",
    "brier_decomposition",
    "brier_score",
    "calibration_report",
    "check_confidence",
    "check_predictions",
    "ece",
    "ece_ci",
    "ece_equal_mass",
    "ece_equal_width",
    "ece_from_confidence",
    "ece_sweep",
    "nll",
    "paired_damage_gap_ci",
    "paired_damage_ratio_ci",
    "paired_delta_accuracy_ci",
    "paired_delta_ece_ci",
    "relative_damages",
    "reliability_curve",
    "reliability_from_confidence",
    "to_quality",
    "top_label",
]
