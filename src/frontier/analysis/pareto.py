"""Pure Pareto-front computation: maximize accuracy, minimize a cost axis.

The variant count is tens, so the O(n^2) pairwise domination test built with numpy
broadcasting is instant and trivially unit-testable. Non-finite points (a missing
cost from a skipped latency profile) are never on the front and never knock a finite
point off it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]


def pareto_mask(accuracy: FloatArray, cost: FloatArray) -> BoolArray:
    """Boolean mask of the Pareto-optimal points: maximize accuracy, minimize cost.

    Point ``i`` is dominated iff some finite ``j`` has ``cost[j] <= cost[i]`` and
    ``accuracy[j] >= accuracy[i]`` with at least one inequality strict; every
    non-dominated finite point gets ``True``. A point with non-finite accuracy or cost
    is never on the front. Co-located or tied points do not dominate each other (the
    strict-inequality requirement), so a set of identical optima all stay on the front.
    """
    if accuracy.shape != cost.shape:
        raise ValueError(f"accuracy has shape {accuracy.shape} but cost has shape {cost.shape}")
    finite = np.isfinite(accuracy) & np.isfinite(cost)
    le_cost = cost[np.newaxis, :] <= cost[:, np.newaxis]
    ge_acc = accuracy[np.newaxis, :] >= accuracy[:, np.newaxis]
    strict = (cost[np.newaxis, :] < cost[:, np.newaxis]) | (
        accuracy[np.newaxis, :] > accuracy[:, np.newaxis]
    )
    dominated = (le_cost & ge_acc & strict & finite[np.newaxis, :]).any(axis=1)
    mask: BoolArray = finite & ~dominated
    return mask


def pareto_order(accuracy: FloatArray, cost: FloatArray) -> IntArray:
    """Indices of the front sorted by cost ascending, ties broken by accuracy descending.

    This is the draw order for the connecting envelope line that traces the
    upper-left frontier.
    """
    front = np.flatnonzero(pareto_mask(accuracy, cost))
    keys = np.lexsort((-accuracy[front], cost[front]))
    ordered: IntArray = front[keys]
    return ordered
