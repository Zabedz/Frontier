"""The bin-free proper scores (NLL, multiclass Brier) and the Murphy decomposition of the
binned Brier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frontier.metrics._array import LabelArray, ProbMatrix, check_predictions

DEFAULT_BINS = 10


def nll(probs: ProbMatrix, gold: LabelArray, *, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the gold class under the answer softmax.

    The gold-class probability is clipped to ``[eps, 1.0]``, so an all-confident-wrong
    forecast scores a finite ``-log(eps)``.
    """
    check_predictions(probs, gold)
    gold_prob = probs[np.arange(gold.shape[0]), gold]
    return float(np.mean(-np.log(np.clip(gold_prob, eps, 1.0))))


def brier_score(probs: ProbMatrix, gold: LabelArray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot gold, in ``[0, 2]``."""
    check_predictions(probs, gold)
    n_items, _ = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n_items), gold] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


@dataclass(frozen=True, slots=True)
class BrierDecomposition:
    """Murphy calibration-refinement decomposition of the binned multiclass Brier.

    ``total == reliability - resolution + uncertainty`` by construction; it is the Brier of
    the bin-mean forecasts, so it meets the raw ``brier_score`` only where each forecast
    already sits at its bin mean.
    """

    reliability: float
    resolution: float
    uncertainty: float
    total: float


def brier_decomposition(
    probs: ProbMatrix, gold: LabelArray, *, n_bins: int = DEFAULT_BINS
) -> BrierDecomposition:
    """Per-class Murphy decomposition summed over classes.

    Class ``k`` is scored as the binary event ``gold == k`` forecast by ``probs[:, k]``,
    binned into ``n_bins`` equal-width bins. With ``n_kb`` the bin count, ``f_kb`` the
    bin-mean forecast, ``o_kb`` the observed frequency and ``o_k`` the base rate:
    reliability is ``(1/N) sum_k sum_b n_kb (f_kb - o_kb)**2``, resolution
    ``(1/N) sum_k sum_b n_kb (o_kb - o_k)**2``, uncertainty ``sum_k o_k (1 - o_k)`` with
    no ``1/N``.
    """
    check_predictions(probs, gold)
    n_items, n_classes = probs.shape
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    uncertainty = 0.0
    for k in range(n_classes):
        forecast = probs[:, k]
        event = (gold == k).astype(np.float64)
        base_rate = float(np.mean(event))
        count = np.histogram(forecast, bins=edges)[0]
        sum_forecast = np.histogram(forecast, bins=edges, weights=forecast)[0]
        sum_event = np.histogram(forecast, bins=edges, weights=event)[0]
        populated = count > 0
        safe_count = np.where(populated, count, 1)
        mean_forecast = np.where(populated, sum_forecast / safe_count, 0.0)
        mean_event = np.where(populated, sum_event / safe_count, 0.0)
        reliability += float(np.sum(count * (mean_forecast - mean_event) ** 2))
        resolution += float(np.sum(count * (mean_event - base_rate) ** 2))
        uncertainty += base_rate * (1.0 - base_rate)
    reliability /= n_items
    resolution /= n_items
    total = reliability - resolution + uncertainty
    return BrierDecomposition(
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        total=total,
    )
