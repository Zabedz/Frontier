"""The held-out split: keyed on the item, so membership survives seeds and subset sizes."""

from __future__ import annotations

import numpy as np
import pytest

from frontier.analysis.holdout import (
    FIT_POSITIONS,
    HOLDOUT_STRIDE,
    NotRecalibratableError,
    fit_mask,
    is_fit,
    split,
    take,
)
from frontier.io.predictions import PredictionRows, pad_option_probs

N_ITEMS = 4000
SMALL = 100
TARGET_FIT_SHARE = FIT_POSITIONS / HOLDOUT_STRIDE


def _rows(qids: list[str], *, seed: int = 0) -> PredictionRows:
    rng = np.random.default_rng(seed)
    draws = [rng.dirichlet(np.ones(4)) for _ in qids]
    predicted = np.asarray([int(draw.argmax()) for draw in draws], dtype=np.intp)
    gold = rng.integers(0, 4, size=len(qids)).astype(np.intp)
    return PredictionRows(
        confidence=np.asarray([float(draw.max()) for draw in draws], dtype=np.float64),
        correct=(predicted == gold),
        gold=gold,
        predicted=predicted,
        options=pad_option_probs([np.asarray(draw, dtype=np.float64) for draw in draws]),
        qid=np.asarray(qids, dtype=np.str_),
    )


def test_membership_is_a_property_of_the_item_not_its_position() -> None:
    """The bug this replaced: an item at a different offset changed sides."""
    ids = [f"q{index}" for index in range(200)]
    forward = fit_mask(np.asarray(ids, dtype=np.str_))
    shuffled = list(reversed(ids))
    backward = fit_mask(np.asarray(shuffled, dtype=np.str_))
    for position, qid in enumerate(shuffled):
        assert backward[position] == forward[ids.index(qid)]


def test_two_overlapping_slices_agree_on_every_shared_item() -> None:
    """Two seeds subsample differently; a shared item must land on one side only."""
    first = [f"q{index}" for index in range(0, 3000, 2)]
    second = [f"q{index}" for index in range(0, 3000, 3)]
    fit_first = {qid for qid in first if is_fit(qid)}
    fit_second = {qid for qid in second if is_fit(qid)}
    shared = set(first) & set(second)
    assert shared
    assert {qid for qid in shared if qid in fit_first} == {
        qid for qid in shared if qid in fit_second
    }


def test_the_fit_share_lands_near_the_committed_fraction() -> None:
    mask = fit_mask(np.asarray([f"q{index}" for index in range(N_ITEMS)], dtype=np.str_))
    assert mask.mean() == pytest.approx(TARGET_FIT_SHARE, abs=0.02)


def test_the_split_is_stable_across_calls() -> None:
    ids = np.asarray([f"q{index}" for index in range(500)], dtype=np.str_)
    assert np.array_equal(fit_mask(ids), fit_mask(ids))


def test_the_halves_are_disjoint_and_cover_every_item() -> None:
    rows = _rows([f"q{index}" for index in range(SMALL)])
    fit, report = split(rows)
    assert fit.qid is not None
    assert report.qid is not None
    assert set(fit.qid).isdisjoint(set(report.qid))
    assert len(set(fit.qid) | set(report.qid)) == SMALL


def test_split_carries_every_column_onto_both_halves() -> None:
    rows = _rows([f"q{index}" for index in range(SMALL)])
    fit, report = split(rows)
    assert fit.options is not None
    assert report.options is not None
    assert fit.gold.shape[0] + report.gold.shape[0] == SMALL
    assert fit.options.probs.shape[0] == fit.gold.shape[0]
    assert fit.options.n_options.shape[0] == fit.gold.shape[0]


def test_split_keeps_each_item_with_its_own_distribution() -> None:
    """A misaligned take would break confidence == probs.max()."""
    rows = _rows([f"q{index}" for index in range(300)])
    for half in split(rows):
        assert half.options is not None
        assert np.allclose(half.options.probs.max(axis=1), half.confidence, atol=0.0)


def test_a_sidecar_with_no_ids_cannot_be_split() -> None:
    rows = _rows([f"q{index}" for index in range(20)])
    with pytest.raises(NotRecalibratableError, match="no qid"):
        split(PredictionRows(rows.confidence, rows.correct, rows.gold, rows.predicted, None, None))


def test_a_slice_too_small_to_fill_both_halves_is_rejected() -> None:
    rows = _rows(["q0"])
    with pytest.raises(NotRecalibratableError, match="half empty"):
        split(rows)


def test_take_on_a_sidecar_without_distributions_stays_none() -> None:
    rows = PredictionRows(
        confidence=np.asarray([0.9, 0.8, 0.7]),
        correct=np.asarray([True, False, True]),
        gold=np.zeros(3, dtype=np.intp),
        predicted=np.zeros(3, dtype=np.intp),
        options=None,
        qid=None,
    )
    taken = take(rows, np.asarray([0, 2], dtype=np.intp))
    assert taken.options is None
    assert taken.qid is None
    assert taken.confidence.shape == (2,)
