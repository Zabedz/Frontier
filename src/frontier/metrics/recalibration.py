"""Temperature scaling over the stored answer-letter distributions.

A single temperature is the cheapest post-hoc repair and the one the compression
literature reports closing most of the calibration gap, so it is the control every
per-family repairability claim is measured against.

Fitting on stored probabilities equals fitting on the original logits: ``log(p) = z - logZ``
and the ``logZ`` is constant across a row, so it cancels in the softmax.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

from frontier.metrics._array import (
    FloatArray,
    IntArray,
    LabelArray,
    ProbMatrix,
    check_predictions,
)

TEMPERATURE_BOUNDS = (0.05, 20.0)

# Wider than the optimiser's own xatol (1e-5), so a run that stopped at a bound is not
# mistaken for one that converged just inside it.
BOUND_ATOL = 1e-3


def _check_options_shape(probs: ProbMatrix, n_options: IntArray) -> None:
    if n_options.shape[0] != probs.shape[0]:
        raise ValueError(f"n_options has {n_options.shape[0]} entries for {probs.shape[0]} items")
    if n_options.size and int(n_options.min()) < 1:
        raise ValueError(f"every item needs at least one option, got {int(n_options.min())}")


def _masked_log_probs(probs: ProbMatrix, n_options: IntArray) -> FloatArray:
    """Row log-probabilities with padding columns at ``-inf`` so they never carry mass."""
    live = np.arange(probs.shape[1])[None, :] < np.asarray(n_options)[:, None]
    with np.errstate(divide="ignore"):
        logged: FloatArray = np.where(live, np.log(probs), -np.inf)
    return logged


def _log_softmax(scores: FloatArray) -> FloatArray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    normalised: FloatArray = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return normalised


def apply_temperature(probs: ProbMatrix, temperature: float, n_options: IntArray) -> ProbMatrix:
    """Re-normalise each row at ``temperature``; padding columns come back 0.0.

    ``temperature`` above 1 softens the distribution, below 1 sharpens it.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    _check_options_shape(probs, n_options)
    scaled: ProbMatrix = np.exp(_log_softmax(_masked_log_probs(probs, n_options) / temperature))
    return scaled


def temperature_nll(
    probs: ProbMatrix, gold: LabelArray, n_options: IntArray, temperature: float
) -> float:
    """Mean negative log-likelihood of ``gold`` under ``probs`` at ``temperature``."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    check_predictions(probs, gold)
    _check_options_shape(probs, n_options)
    log_probs = _log_softmax(_masked_log_probs(probs, n_options) / temperature)
    return float(-np.mean(log_probs[np.arange(gold.shape[0]), gold]))


def fit_temperature(
    probs: ProbMatrix,
    gold: LabelArray,
    n_options: IntArray,
    *,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
) -> float:
    """The temperature minimising held-out NLL.

    Fit on a held-out split. Fitting and reporting on the same items understates the
    residual calibration error, which is the quantity the repairability claim rests on.

    Raises ``ValueError`` when the optimiser fails, when the objective is not finite (one
    item with zero mass on its gold class sends the mean NLL to infinity at every
    temperature), or when the minimum sits at a bound. A fit that stopped at a bound is a
    sample whose optimum lies outside the scalable range, and returning the bound would
    read as convergence.
    """
    check_predictions(probs, gold)
    _check_options_shape(probs, n_options)
    log_probs = _masked_log_probs(probs, n_options)
    rows = np.arange(gold.shape[0])

    def objective(temperature: float) -> float:
        scaled = _log_softmax(log_probs / temperature)
        return float(-np.mean(scaled[rows, gold]))

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    best = float(result.x)
    if not bool(result.success) or not math.isfinite(float(result.fun)):
        raise ValueError(
            f"temperature fit did not converge: {result.message}, T={best}, nll={result.fun}"
        )
    low, high = bounds
    if best - low < BOUND_ATOL or high - best < BOUND_ATOL:
        raise ValueError(
            f"temperature fit ran to the bound {bounds}: T={best}. The sample's optimum "
            f"lies outside the scalable range."
        )
    return best
