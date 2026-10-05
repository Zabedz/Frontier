"""The paired difference of two shifts, on hand-built rows where the answer is known."""

from __future__ import annotations

import numpy as np
import pytest

from frontier.metrics._array import CorrectArray, FloatArray
from frontier.metrics.bootstrap import paired_shift_difference_ci
from frontier.metrics.calibration import ece_from_confidence

N_ITEMS = 400
SHIFT = 0.05
RESAMPLES = 199


def _row(seed: int) -> tuple[FloatArray, CorrectArray]:
    rng = np.random.default_rng(seed)
    confidence: FloatArray = rng.uniform(0.3, 0.9, N_ITEMS)
    correct: CorrectArray = rng.random(N_ITEMS) < confidence
    return confidence, correct


def test_a_constant_confidence_shift_is_recovered_exactly() -> None:
    confidence, correct = _row(0)
    interval = paired_shift_difference_ci(
        [confidence, confidence + SHIFT, confidence, confidence],
        [correct] * 4,
        metric="confidence",
        n_resamples=RESAMPLES,
        rng=0,
    )
    assert interval.point == pytest.approx(SHIFT)
    assert interval.low == pytest.approx(SHIFT)
    assert interval.high == pytest.approx(SHIFT)
    assert interval.excludes_zero


def test_swapping_the_methods_flips_the_sign() -> None:
    confidence, correct = _row(0)
    interval = paired_shift_difference_ci(
        [confidence, confidence, confidence, confidence + SHIFT],
        [correct] * 4,
        metric="confidence",
        n_resamples=RESAMPLES,
        rng=0,
    )
    assert interval.point == pytest.approx(-SHIFT)


def test_ece_point_is_the_difference_of_the_two_shifts() -> None:
    rows = [_row(seed) for seed in range(4)]
    interval = paired_shift_difference_ci(
        [conf for conf, _ in rows],
        [corr for _, corr in rows],
        metric="ece",
        n_resamples=RESAMPLES,
        rng=0,
    )
    ece = [ece_from_confidence(conf, corr) for conf, corr in rows]
    assert interval.point == pytest.approx((ece[1] - ece[0]) - (ece[3] - ece[2]))
    assert interval.low <= interval.point <= interval.high


def test_identical_shifts_cancel_in_every_resample() -> None:
    """Exactly zero throughout only if one index vector resamples all eight arrays."""
    before, before_correct = _row(0)
    after, after_correct = _row(1)
    interval = paired_shift_difference_ci(
        [before, after, before, after],
        [before_correct, after_correct, before_correct, after_correct],
        metric="ece",
        n_resamples=RESAMPLES,
        rng=0,
    )
    assert interval.point == 0.0
    assert interval.low == 0.0
    assert interval.high == 0.0


def test_refuses_a_row_count_other_than_four() -> None:
    confidence, correct = _row(0)
    with pytest.raises(ValueError, match="needs 4 rows"):
        paired_shift_difference_ci(
            [confidence] * 3, [correct] * 3, metric="confidence", n_resamples=RESAMPLES
        )


def test_refuses_rows_over_different_items() -> None:
    confidence, correct = _row(0)
    with pytest.raises(ValueError, match="same items"):
        paired_shift_difference_ci(
            [confidence, confidence, confidence, confidence[:-1]],
            [correct] * 4,
            metric="confidence",
            n_resamples=RESAMPLES,
        )
