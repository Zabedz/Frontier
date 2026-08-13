"""Equal-width and equal-mass binning of top-label confidence, shared by the ECE family
and the Brier decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from frontier.metrics._array import CorrectArray, FloatArray, IntArray

BinScheme = Literal["equal_width", "equal_mass"]


@dataclass(frozen=True, slots=True)
class BinStats:
    """Per-bin counts and means over a single confidence array.

    The zeroed means on an empty bin keep it out of both the mass-weighted sum and the
    equal-weight average.
    """

    edges: FloatArray  # (n_effective_bins + 1,)
    count: IntArray  # (n_effective_bins,)
    mean_confidence: FloatArray  # 0.0 where count == 0
    accuracy: FloatArray  # 0.0 where count == 0


def bin_edges(confidence: FloatArray, n_bins: int, scheme: BinScheme) -> FloatArray:
    """Return the bin edges for ``scheme`` over ``[0, 1]``.

    Equal-mass edges are confidence quantiles, de-duplicated: a near-constant confidence
    distribution supports fewer than ``n_bins`` distinct cuts, and the effective bin count
    falls to what the quantiles can carry.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if scheme == "equal_width":
        return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    edges = np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1)).astype(np.float64)
    edges[0] = 0.0
    edges[-1] = 1.0
    deduped: FloatArray = np.unique(edges)
    return deduped


def bin_stats(
    confidence: FloatArray, correct: CorrectArray, n_bins: int, scheme: BinScheme
) -> BinStats:
    """Bin ``confidence`` and reduce ``correct`` within each bin.

    ``np.histogram`` does the assignment, which matches netcal exactly and torchmetrics on
    interior confidences.
    """
    edges = bin_edges(confidence, n_bins, scheme)
    count = np.histogram(confidence, bins=edges)[0]
    sum_confidence = np.histogram(confidence, bins=edges, weights=confidence)[0]
    sum_correct = np.histogram(confidence, bins=edges, weights=correct.astype(np.float64))[0]
    populated = count > 0
    safe_count = np.where(populated, count, 1)
    mean_confidence = np.where(populated, sum_confidence / safe_count, 0.0)
    accuracy = np.where(populated, sum_correct / safe_count, 0.0)
    return BinStats(
        edges=edges,
        count=count.astype(np.intp),
        mean_confidence=mean_confidence,
        accuracy=accuracy,
    )
