"""Reductions on synthetic samples with hand-computed answers, no torch, no network."""

from __future__ import annotations

import math

import pytest

from frontier.latency.stats import (
    TrialTiming,
    discard_warmup,
    percentile,
    reduce_trials,
)


def test_median_of_one_to_twenty() -> None:
    assert percentile(list(range(1, 21)), 50.0) == pytest.approx(10.5)


def test_p95_linear_method_known_answer() -> None:
    # Linear method on n=20: rank 19 * 0.95 = 18.05, so 19 + 0.05 * (20 - 19).
    assert percentile(list(range(1, 21)), 95.0) == pytest.approx(19.05)


def test_singleton_median_equals_p95() -> None:
    assert percentile([7.5], 50.0) == pytest.approx(7.5)
    assert percentile([7.5], 95.0) == pytest.approx(7.5)


def test_percentile_of_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50.0)


def test_discard_warmup_drops_the_first_k() -> None:
    trials = [TrialTiming(float(i), (float(i),)) for i in range(5)]
    kept = discard_warmup(trials, 2)
    assert [trial.ttft_ms for trial in kept] == [2.0, 3.0, 4.0]


def test_discard_warmup_beyond_length_empties() -> None:
    trials = [TrialTiming(1.0, (1.0,)), TrialTiming(2.0, (2.0,))]
    assert discard_warmup(trials, 5) == []


def test_reduce_trials_known_answers() -> None:
    trials = [
        TrialTiming(1.0, (1.0,)),
        TrialTiming(10.0, (2.0, 4.0)),
        TrialTiming(20.0, (6.0,)),
        TrialTiming(30.0, (8.0,)),
    ]
    stats = reduce_trials(trials, batch_size=4, warmup=1)

    assert stats.n_trials == len(trials) - 1
    assert stats.warmup_discarded == 1
    assert stats.ttft_median_ms == pytest.approx(20.0)
    assert stats.ttft_p95_ms == pytest.approx(29.0)
    assert stats.itl_median_ms == pytest.approx(5.0)
    assert stats.itl_p95_ms == pytest.approx(7.7)
    # throughput = batch_size * 1000 / itl_median = 4 * 1000 / 5.
    assert stats.throughput_tok_s == pytest.approx(800.0)


def test_reduce_trials_no_survivors_raises() -> None:
    trials = [TrialTiming(1.0, (1.0,)), TrialTiming(2.0, (2.0,))]
    with pytest.raises(ValueError, match="survive"):
        reduce_trials(trials, batch_size=1, warmup=2)


def test_reduce_trials_empty_pooled_itl_is_nan() -> None:
    trials = [TrialTiming(1.0, ()), TrialTiming(2.0, ())]
    stats = reduce_trials(trials, batch_size=1, warmup=0)
    assert math.isnan(stats.itl_median_ms)
    assert math.isnan(stats.itl_p95_ms)
    assert math.isnan(stats.throughput_tok_s)
    assert stats.ttft_median_ms == pytest.approx(1.5)
