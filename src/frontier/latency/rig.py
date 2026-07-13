"""Orchestration: the batch x context sweep, and the real-probe wiring.

``measure_latency_memory`` runs the sweep over injected seams (a driver, a clock
factory, a machine probe, a memory probe), so the whole orchestration is exercised on
CPU with fakes. ``default_latency`` wires the real probes from a loaded provider and
picks the smoke vs full parameters. Latency is a property of the variant and the
hardware, not of an eval seed, so the runner calls this once per run.

The cost proxy is decimal throughout: MB is bytes / 1e6 and GB is MB / 1000, matching
``docs/results_schema.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

from frontier.eval.provider import LogitProvider
from frontier.latency.machine import (
    MachineProbe,
    NvidiaSmiProbe,
    probe_clock_lock,
    to_machine_state,
)
from frontier.latency.memory import HFMemoryProbe, MemoryProbe
from frontier.latency.stats import LatencyStats, reduce_trials
from frontier.latency.timing import (
    CudaEventClock,
    GenerationDriver,
    HFGenerationDriver,
    PerfCounterClock,
    TrialClock,
    collect_trials,
)
from frontier.schema import Latency, LatencySpec, Memory, RunMode

if TYPE_CHECKING:
    from frontier.pipeline.config import ResolvedConfig

FULL_DECODE_LEN = 64
SMOKE_DECODE_LEN = 8
_MB_PER_GB = 1000.0


@dataclass(frozen=True, slots=True)
class LatencyMemory:
    """The three fields the rig fills on every ``ResultRow``."""

    latency: list[Latency]
    memory: list[Memory]
    tok_s_per_gb: float


def measure_latency_memory(
    driver: GenerationDriver,
    spec: LatencySpec,
    *,
    decode_len: int,
    clock_factory: Callable[[], TrialClock],
    machine: MachineProbe,
    memory_probe: MemoryProbe,
    clocks_locked: bool,
    reference_batch: int | None = None,
) -> LatencyMemory:
    """Run the batch x context sweep and assemble the schema records.

    Per ``bs`` in ``spec.batch_sizes``: capture ``before`` clocks, collect
    ``warmup + n_trials`` trials at the fixed prefill length ``context_lengths[0]``,
    capture ``after`` clocks, reduce, and build one ``Latency`` whose ``machine_state``
    is ``to_machine_state(before, after)``. TTFT is measured at that one prefill length
    because ``schema.Latency`` has no per-context slot; ``Memory`` carries the full
    context sweep, one entry per ``(bs, ctx)``.

    The cost proxy uses ``reference_batch`` (default ``max(batch_sizes)``): its
    throughput over its peak VRAM at the reference context, ``NaN`` when that peak is
    ``<= 0``.
    """
    context_len = spec.context_lengths[0]
    n_total = spec.warmup + spec.n_trials
    latency: list[Latency] = []
    stats_by_batch: dict[int, LatencyStats] = {}
    for batch_size in spec.batch_sizes:
        before = machine.capture()
        trials = collect_trials(
            driver,
            clock_factory,
            batch_size=batch_size,
            context_len=context_len,
            decode_len=decode_len,
            n_total=n_total,
        )
        after = machine.capture()
        stats = reduce_trials(trials, batch_size=batch_size, warmup=spec.warmup)
        stats_by_batch[batch_size] = stats
        latency.append(
            Latency(
                batch_size=batch_size,
                ttft_median_ms=stats.ttft_median_ms,
                ttft_p95_ms=stats.ttft_p95_ms,
                itl_median_ms=stats.itl_median_ms,
                itl_p95_ms=stats.itl_p95_ms,
                throughput_tok_s=stats.throughput_tok_s,
                n_trials=stats.n_trials,
                warmup_discarded=stats.warmup_discarded,
                machine_state=to_machine_state(before, after, clocks_locked=clocks_locked),
            )
        )

    memory: list[Memory] = []
    peak_by_key: dict[tuple[int, int], float] = {}
    for batch_size, context in product(spec.batch_sizes, spec.context_lengths):
        entry = memory_probe.measure(batch_size=batch_size, context_len=context)
        memory.append(entry)
        peak_by_key[(batch_size, context)] = entry.peak_vram_mb

    reference = reference_batch if reference_batch is not None else max(spec.batch_sizes)
    tok_s_per_gb = _cost_proxy(
        stats_by_batch[reference].throughput_tok_s, peak_by_key[(reference, context_len)]
    )
    return LatencyMemory(latency=latency, memory=memory, tok_s_per_gb=tok_s_per_gb)


def default_latency(
    provider: LogitProvider,
    resolved: ResolvedConfig,
    *,
    device: str,
    mode: RunMode,
) -> LatencyMemory:
    """Wire the real probes from a loaded provider and run the sweep.

    Requires ``provider.loaded_model`` (only the HF Track-A provider is in scope);
    raises ``ValueError`` naming the provider type otherwise. Builds an
    ``HFGenerationDriver`` and an ``HFMemoryProbe`` from the loaded model, an
    ``NvidiaSmiProbe`` for the machine state, and the clock factory for the device.
    ``decode_len`` is ``SMOKE_DECODE_LEN`` in smoke and ``FULL_DECODE_LEN`` in full; the
    batch/context/n_trials/warmup all come from the resolved ``LatencySpec``.
    """
    load = getattr(provider, "loaded_model", None)
    if load is None:
        raise ValueError(
            f"latency rig needs a provider with loaded_model(); "
            f"{type(provider).__name__} has none (only the HF Track-A provider is in scope)"
        )
    model, tokenizer = load()
    torch_module = _import_torch()
    decode_len = SMOKE_DECODE_LEN if mode == "smoke" else FULL_DECODE_LEN
    driver = HFGenerationDriver(model, tokenizer, device)
    memory_probe = HFMemoryProbe(
        driver,
        model,
        device=device,
        decode_len=decode_len,
        model_dir=_snapshot_dir(resolved),
        torch_module=torch_module,
    )
    return measure_latency_memory(
        driver,
        resolved.variant.latency,
        decode_len=decode_len,
        clock_factory=_clock_factory(device, torch_module),
        machine=NvidiaSmiProbe(),
        memory_probe=memory_probe,
        clocks_locked=probe_clock_lock(),
    )


def _cost_proxy(throughput_tok_s: float, peak_vram_mb: float) -> float:
    if peak_vram_mb <= 0.0:
        return math.nan
    return throughput_tok_s / (peak_vram_mb / _MB_PER_GB)


def _clock_factory(device: str, torch_module: Any) -> Callable[[], TrialClock]:
    if device.startswith("cuda"):
        return lambda: CudaEventClock(torch_module)  # pragma: no cover
    return PerfCounterClock


def _import_torch() -> Any:
    import torch  # noqa: PLC0415

    return torch


def _snapshot_dir(resolved: ResolvedConfig) -> Path | None:
    """The local HF cache snapshot dir for the variant's model, or ``None`` if uncached.

    Resolved through ``huggingface_hub.try_to_load_from_cache`` on ``config.json``, which
    never hits the network. A cache miss or a missing ``huggingface_hub`` yields ``None``,
    so ``weights_disk_mb`` degrades to ``0.0`` rather than raising.
    """
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    except ImportError:
        return None
    cached = try_to_load_from_cache(
        resolved.variant.model.model_id,
        "config.json",
        revision=resolved.variant.model.model_revision,
    )
    return Path(cached).parent if isinstance(cached, str) else None
