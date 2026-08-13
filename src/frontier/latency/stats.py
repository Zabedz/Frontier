"""Pure numpy reductions over timing samples: percentiles, warmup discard, aggregation.

TTFT and inter-token latency stay in separate columns through the reduction, and both
reduce by median and p95 because the decode-step distribution is skewed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

MEDIAN_Q = 50.0
P95_Q = 95.0
MS_PER_S = 1000.0


@dataclass(frozen=True, slots=True)
class TrialTiming:
    """One decode trial: its TTFT and its per-token ITL spans."""

    ttft_ms: float
    itl_ms: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """The reduced timing summary for one batch size, before it becomes a ``Latency``."""

    ttft_median_ms: float
    ttft_p95_ms: float
    itl_median_ms: float
    itl_p95_ms: float
    throughput_tok_s: float
    n_trials: int
    warmup_discarded: int


def percentile(samples: Sequence[float], q: float) -> float:
    """The ``q``th percentile under numpy's ``method="linear"``, fixed for exact answers."""
    if len(samples) == 0:
        raise ValueError(f"cannot take the {q} percentile of an empty timing sample")
    return float(np.percentile(np.asarray(samples, dtype=float), q, method="linear"))


def discard_warmup(trials: Sequence[TrialTiming], warmup: int) -> list[TrialTiming]:
    """Drop the first ``min(warmup, len(trials))`` trials, so ``warmup >= len`` empties."""
    return list(trials[min(max(warmup, 0), len(trials)) :])


def reduce_trials(trials: Sequence[TrialTiming], *, batch_size: int, warmup: int) -> LatencyStats:
    """Discard warmup, then reduce the survivors to one ``LatencyStats``.

    TTFT contributes one sample per kept trial; ITL pools every per-token span across all
    kept trials. ``throughput_tok_s`` is ``batch_size * 1000 / itl_median_ms``, the
    sustained decode rate at the reported median ITL, and goes ``NaN`` when the pooled ITL
    is empty or its median is zero.
    """
    kept = discard_warmup(trials, warmup)
    discarded = len(trials) - len(kept)
    if not kept:
        raise ValueError(f"no trials survive discarding {warmup} warmup from {len(trials)} total")
    ttft = [trial.ttft_ms for trial in kept]
    itl = [span for trial in kept for span in trial.itl_ms]
    if itl:
        itl_median = percentile(itl, MEDIAN_Q)
        itl_p95 = percentile(itl, P95_Q)
        throughput = batch_size * MS_PER_S / itl_median if itl_median > 0 else math.nan
    else:
        itl_median = math.nan
        itl_p95 = math.nan
        throughput = math.nan
    return LatencyStats(
        ttft_median_ms=percentile(ttft, MEDIAN_Q),
        ttft_p95_ms=percentile(ttft, P95_Q),
        itl_median_ms=itl_median,
        itl_p95_ms=itl_p95,
        throughput_tok_s=throughput,
        n_trials=len(kept),
        warmup_discarded=discarded,
    )
