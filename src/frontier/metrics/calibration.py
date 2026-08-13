"""Top-label reduction and the ECE family. Every entry point and the paired bootstrap reduce
to ``ece_from_confidence``, which takes 1-D arrays a paired resample can index with one
vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from frontier.metrics._array import (
    CorrectArray,
    FloatArray,
    IntArray,
    LabelArray,
    ProbMatrix,
    check_confidence,
    check_predictions,
)
from frontier.metrics.binning import BinScheme, bin_stats

Weighting = Literal["mass", "equal"]

DEFAULT_BINS = 10
DEFAULT_SWEEP = (5, 10, 15, 20, 30, 50)


def top_label(probs: ProbMatrix, gold: LabelArray) -> tuple[FloatArray, CorrectArray]:
    """Reduce a softmax to its top-label confidence and correctness."""
    confidence: FloatArray = probs.max(axis=1)
    correct: CorrectArray = probs.argmax(axis=1) == gold
    return confidence, correct


def ece_from_confidence(
    confidence: FloatArray,
    correct: CorrectArray,
    *,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
    weighting: Weighting = "mass",
) -> float:
    """Expected calibration error from reduced confidence and correctness.

    ``weighting="mass"`` is the classic bin-mass-weighted ECE; ``"equal"`` averages the
    deviation over populated bins only (the ACE aggregation).
    """
    check_confidence(confidence, correct)
    stats = bin_stats(confidence, correct, n_bins, scheme)
    deviation = np.abs(stats.accuracy - stats.mean_confidence)
    if weighting == "mass":
        total = confidence.shape[0]
        return float(np.sum(stats.count / total * deviation))
    populated = stats.count > 0
    return float(np.mean(deviation[populated]))


def ece(
    probs: ProbMatrix,
    gold: LabelArray,
    *,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
    weighting: Weighting = "mass",
) -> float:
    """Expected calibration error over the top-label softmax."""
    check_predictions(probs, gold)
    confidence, correct = top_label(probs, gold)
    return ece_from_confidence(
        confidence, correct, n_bins=n_bins, scheme=scheme, weighting=weighting
    )


def ece_equal_width(probs: ProbMatrix, gold: LabelArray, *, n_bins: int = DEFAULT_BINS) -> float:
    """Mass-weighted equal-width ECE (Guo/Naeini), the reference calibration error."""
    return ece(probs, gold, n_bins=n_bins, scheme="equal_width", weighting="mass")


def ece_equal_mass(probs: ProbMatrix, gold: LabelArray, *, n_bins: int = DEFAULT_BINS) -> float:
    """Mass-weighted equal-mass (adaptive) ECE, the lower-bias companion.

    Under equal bin masses this coincides with the equal-weight ACE average, so it is what
    populates ``Quality.ece_equal_mass_ace``.
    """
    return ece(probs, gold, n_bins=n_bins, scheme="equal_mass", weighting="mass")


def ece_sweep(
    probs: ProbMatrix,
    gold: LabelArray,
    *,
    bin_counts: tuple[int, ...] = DEFAULT_SWEEP,
    scheme: BinScheme = "equal_width",
) -> dict[int, float]:
    """Map each bin count to its ECE, in the shape ``Quality.ece_bin_sweep`` takes."""
    check_predictions(probs, gold)
    confidence, correct = top_label(probs, gold)
    return {
        n: ece_from_confidence(confidence, correct, n_bins=n, scheme=scheme, weighting="mass")
        for n in bin_counts
    }


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    """Per-bin reliability-diagram data."""

    edges: FloatArray
    mean_confidence: FloatArray  # np.nan where the bin is empty
    accuracy: FloatArray  # np.nan where the bin is empty
    count: IntArray


def reliability_from_confidence(
    confidence: FloatArray, correct: CorrectArray, n_bins: int, scheme: BinScheme
) -> ReliabilityCurve:
    """Reliability-diagram data from confidence and correctness already reduced."""
    stats = bin_stats(confidence, correct, n_bins, scheme)
    populated = stats.count > 0
    return ReliabilityCurve(
        edges=stats.edges,
        mean_confidence=np.where(populated, stats.mean_confidence, np.nan),
        accuracy=np.where(populated, stats.accuracy, np.nan),
        count=stats.count,
    )


def reliability_curve(
    probs: ProbMatrix,
    gold: LabelArray,
    *,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
) -> ReliabilityCurve:
    """Reliability-diagram data for the top-label softmax.

    Empty bins come back as ``np.nan`` so plotting can tell them from a bin whose accuracy
    is genuinely zero.
    """
    check_predictions(probs, gold)
    confidence, correct = top_label(probs, gold)
    return reliability_from_confidence(confidence, correct, n_bins, scheme)
