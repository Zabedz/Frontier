"""The sweep orchestration over fakes: shapes, the reference batch, the cost proxy."""

from __future__ import annotations

import math

import pytest

from frontier.latency.machine import ClockReading
from frontier.latency.rig import LatencyMemory, measure_latency_memory
from frontier.latency.timing import TrialClock
from frontier.schema import LatencySpec, MachineState, Memory

_SCRIPTED_SPANS = [5.0, 2.0, 4.0]  # one TTFT then two ITL spans per trial
_ITL_MEDIAN_MS = 3.0  # median of the two pooled ITL spans across trials
_CONSTANT_PEAK_MB = 2000.0
_MB_PER_GB = 1000.0

BATCH_SIZES = (1, 4)
CONTEXT_LENGTHS = (128, 256)
N_TRIALS = 2
WARMUP = 1
SM_MHZ = 1500
LARGEST_BATCH = 4


class _ScriptedClock:
    def mark(self) -> None:
        return None

    def resolve(self) -> list[float]:
        return list(_SCRIPTED_SPANS)


class _FakeDriver:
    def run_trial(
        self, clock: TrialClock, *, batch_size: int, context_len: int, decode_len: int
    ) -> None:
        del batch_size, context_len, decode_len
        clock.mark()


class _FakeMachine:
    def capture(self) -> ClockReading:
        return ClockReading(SM_MHZ, 6000, 60, 100.0, present=True)


class _FakeMemory:
    def measure(self, *, batch_size: int, context_len: int) -> Memory:
        return Memory(
            batch_size=batch_size,
            context_len=context_len,
            peak_vram_mb=_CONSTANT_PEAK_MB,
            weights_disk_mb=3000.0,
            weights_resident_mb=3200.0,
            kv_cache_mb=42.0,
        )


def _spec() -> LatencySpec:
    return LatencySpec(
        batch_sizes=BATCH_SIZES, context_lengths=CONTEXT_LENGTHS, n_trials=N_TRIALS, warmup=WARMUP
    )


def _run(reference_batch: int | None = None) -> LatencyMemory:
    return measure_latency_memory(
        _FakeDriver(),
        _spec(),
        decode_len=8,
        clock_factory=_ScriptedClock,
        machine=_FakeMachine(),
        memory_probe=_FakeMemory(),
        clocks_locked=False,
        reference_batch=reference_batch,
    )


def _expected_proxy(batch: int) -> float:
    throughput = batch * 1000.0 / _ITL_MEDIAN_MS
    return throughput / (_CONSTANT_PEAK_MB / _MB_PER_GB)


def test_sweep_shapes_and_machine_state() -> None:
    result = _run()
    assert len(result.latency) == len(BATCH_SIZES)
    assert len(result.memory) == len(BATCH_SIZES) * len(CONTEXT_LENGTHS)
    for entry in result.latency:
        assert isinstance(entry.machine_state, MachineState)
        assert entry.machine_state.gpu_clock_sm_mhz == SM_MHZ
        assert entry.n_trials == N_TRIALS
        assert entry.warmup_discarded == WARMUP


def test_cost_proxy_uses_the_largest_batch_by_default() -> None:
    result = _run()
    assert math.isfinite(result.tok_s_per_gb)
    assert result.tok_s_per_gb == pytest.approx(_expected_proxy(LARGEST_BATCH))


def test_reference_batch_override_changes_the_proxy() -> None:
    result = _run(reference_batch=1)
    assert result.tok_s_per_gb == pytest.approx(_expected_proxy(1))
