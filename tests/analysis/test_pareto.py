"""Known-answer Pareto fronts: maximize accuracy, minimize cost.

The six enumerated corners pin the domination rule exactly, including the tie and
non-finite cases where an off-by-one in the strict-inequality logic would show.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from frontier.analysis.pareto import pareto_mask, pareto_order

FloatArray = npt.NDArray[np.float64]


def _f(values: list[float]) -> FloatArray:
    return np.array(values, dtype=np.float64)


def test_single_point_is_on_the_front() -> None:
    assert pareto_mask(_f([0.8]), _f([100.0])).tolist() == [True]
    assert pareto_order(_f([0.8]), _f([100.0])).tolist() == [0]


def test_strict_dominator_knocks_out_the_dominated() -> None:
    accuracy = _f([0.9, 0.7])
    cost = _f([100.0, 200.0])
    assert pareto_mask(accuracy, cost).tolist() == [True, False]
    assert pareto_order(accuracy, cost).tolist() == [0]


def test_staircase_front_with_interior_dominated_points() -> None:
    # P0..P2 each trade accuracy for cost (all on the front); P3 and P4 sit inside it.
    accuracy = _f([0.9, 0.8, 0.7, 0.75, 0.65])
    cost = _f([300.0, 200.0, 100.0, 250.0, 150.0])
    assert pareto_mask(accuracy, cost).tolist() == [True, True, True, False, False]
    # Drawn cheapest-first: P2 (100), then P1 (200), then P0 (300).
    assert pareto_order(accuracy, cost).tolist() == [2, 1, 0]


def test_identical_points_both_stay_on_the_front() -> None:
    accuracy = _f([0.8, 0.8])
    cost = _f([100.0, 100.0])
    assert pareto_mask(accuracy, cost).tolist() == [True, True]


def test_tied_cost_higher_accuracy_dominates() -> None:
    accuracy = _f([0.9, 0.8])
    cost = _f([100.0, 100.0])
    assert pareto_mask(accuracy, cost).tolist() == [True, False]


def test_nan_cost_is_off_the_front_and_spares_finite_points() -> None:
    accuracy = _f([0.9, 0.8, 0.95])
    cost = _f([100.0, 200.0, np.nan])
    # The NaN point has the highest accuracy but a missing cost: off the front, and it
    # neither dominates nor drops the finite winner P0.
    assert pareto_mask(accuracy, cost).tolist() == [True, False, False]


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        pareto_mask(_f([0.8, 0.7]), _f([100.0]))
