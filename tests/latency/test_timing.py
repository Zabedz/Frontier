"""The clock seam and the trial collector on CPU, plus the stubbed CUDA-event path.

The stub pins methodology section 4 without a GPU: the per-token loop never synchronises,
and ``resolve`` synchronises exactly once.
"""

from __future__ import annotations

import math

import pytest

from frontier.latency.timing import (
    CudaEventClock,
    PerfCounterClock,
    TrialClock,
    collect_trials,
)

N_MARKS = 3
N_TOTAL = 3
DECODE_LEN = 8


class _MockDriver:
    """Marks the clock ``decode_len + 1`` times with negligible work, no model."""

    def run_trial(
        self, clock: TrialClock, *, batch_size: int, context_len: int, decode_len: int
    ) -> None:
        del batch_size, context_len
        for _ in range(decode_len + 1):
            clock.mark()


class _TooFewMarksDriver:
    """Marks twice, so a trial yields one span and ``collect_trials`` rejects it."""

    def run_trial(
        self, clock: TrialClock, *, batch_size: int, context_len: int, decode_len: int
    ) -> None:
        del batch_size, context_len, decode_len
        clock.mark()
        clock.mark()


class _FakeEvent:
    def __init__(self, source: _FakeCuda) -> None:
        self._source = source
        self._stamp = 0.0

    def record(self) -> None:
        self._stamp = self._source.tick()

    def elapsed_time(self, other: _FakeEvent) -> float:
        return other._stamp - self._stamp


class _FakeCuda:
    def __init__(self) -> None:
        self.sync_calls = 0
        self._time = 0.0

    def tick(self) -> float:
        self._time += 1.0
        return self._time

    def Event(self, *, enable_timing: bool) -> _FakeEvent:  # noqa: N802
        assert enable_timing
        return _FakeEvent(self)

    def synchronize(self) -> None:
        self.sync_calls += 1


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


def test_perf_counter_clock_spans_are_finite_and_non_negative() -> None:
    clock = PerfCounterClock()
    for _ in range(N_MARKS):
        clock.mark()
    spans = clock.resolve()
    assert len(spans) == N_MARKS - 1
    assert all(math.isfinite(span) and span >= 0.0 for span in spans)


def test_collect_trials_separates_ttft_from_itl() -> None:
    trials = collect_trials(
        _MockDriver(),
        PerfCounterClock,
        batch_size=1,
        context_len=4,
        decode_len=DECODE_LEN,
        n_total=N_TOTAL,
    )
    assert len(trials) == N_TOTAL
    for trial in trials:
        assert len(trial.itl_ms) == DECODE_LEN - 1


def test_collect_trials_rejects_a_trial_with_too_few_spans() -> None:
    with pytest.raises(ValueError, match="ITL sample"):
        collect_trials(
            _TooFewMarksDriver(),
            PerfCounterClock,
            batch_size=1,
            context_len=4,
            decode_len=DECODE_LEN,
            n_total=1,
        )


def test_cuda_event_clock_synchronises_once_and_never_mid_loop() -> None:
    torch = _FakeTorch()
    clock = CudaEventClock(torch)
    for _ in range(N_MARKS):
        clock.mark()
    assert torch.cuda.sync_calls == 0

    spans = clock.resolve()
    assert torch.cuda.sync_calls == 1
    assert spans == [1.0, 1.0]
