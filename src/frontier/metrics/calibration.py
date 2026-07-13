"""Top-label reduction and the ECE family.

``ece_from_confidence`` is the load-bearing core; every ECE entry point and the
paired bootstrap reduce to it. It works only on the 1-D confidence and correctness
arrays, never on the softmax, so a paired resample indexes it with one vector.
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
    """Reduce a softmax to its top-label confidence and correctness.

    Returns the per-item max softmax probability and whether the argmax class is
    the gold class.
    """
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

    ``weighting="mass"`` is the classic bin-mass-weighted ECE; ``"equal"`` averages
    the per-bin deviation over populated bins only (the ACE aggregation).
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

    Under equal bin masses the mass weighting coincides with the equal-weight ACE
    average; this value populates ``Quality.ece_equal_mass_ace``.
    """
    return ece(probs, gold, n_bins=n_bins, scheme="equal_mass", weighting="mass")


def ece_sweep(
    probs: ProbMatrix,
    gold: LabelArray,
    *,
    bin_counts: tuple[int, ...] = DEFAULT_SWEEP,
    scheme: BinScheme = "equal_width",
) -> dict[int, float]:
    """Map each bin count to its ECE, for the bin-sensitivity sweep.

    The keys and value type match ``Quality.ece_bin_sweep``.
    """
    check_predictions(probs, gold)
    confidence, correct = top_label(probs, gold)
    return {
        n: ece_from_confidence(confidence, correct, n_bins=n, scheme=scheme, weighting="mass")
        for n in bin_counts
    }


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    """Per-bin reliability-diagram data, empty bins carried as ``np.nan``.

    ``mean_confidence`` and ``accuracy`` are ``np.nan`` where the bin is empty, so
    a later plotting step drops empty bins rather than drawing them at the origin.
    """

    edges: FloatArray
    mean_confidence: FloatArray  # np.nan where the bin is empty
    accuracy: FloatArray  # np.nan where the bin is empty
    count: IntArray


def reliability_from_confidence(
    confidence: FloatArray, correct: CorrectArray, n_bins: int, scheme: BinScheme
) -> ReliabilityCurve:
    """Reliability-diagram data from the reduced confidence and correctness.

    The confidence-space core of ``reliability_curve``, so a caller that has
    already reduced the softmax reuses it rather than reducing twice.
    """
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

    A separate output from the ECE path: empty bins are carried as ``np.nan`` here
    rather than zeroed, since plotting needs to distinguish an empty bin from a bin
    whose accuracy is genuinely zero.
    """
    check_predictions(probs, gold)
    confidence, correct = top_label(probs, gold)
    return reliability_from_confidence(confidence, correct, n_bins, scheme)
