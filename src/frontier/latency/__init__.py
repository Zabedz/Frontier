"""Latency, memory, and machine-state capture: TTFT and inter-token latency timed
separately with CUDA events, warmup discarded, median and p95 over repeated trials;
peak VRAM and KV-cache growth; and per-measurement GPU clock, temperature, and power
so latency numbers are defensible on a machine whose clocks cannot be locked.

The orchestration entry points are re-exported here; the seams (clocks, drivers,
probes) live in the submodules and are imported directly where they are wired.
"""

from frontier.latency.rig import (
    FULL_DECODE_LEN,
    SMOKE_DECODE_LEN,
    LatencyMemory,
    default_latency,
    measure_latency_memory,
)

__all__ = [
    "FULL_DECODE_LEN",
    "SMOKE_DECODE_LEN",
    "LatencyMemory",
    "default_latency",
    "measure_latency_memory",
]
