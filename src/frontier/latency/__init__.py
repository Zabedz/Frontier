"""Latency, memory, and machine-state capture for one run.

The orchestration entry points are re-exported from ``rig``; the seams stay in the
submodules and are imported directly where they are wired.
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
