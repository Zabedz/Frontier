"""Track-B latency from each serving stack's own benchmarker.

The WP4 CUDA-event rig drives an ``nn.Module`` token by token; vLLM and llama.cpp are not
``nn.Module``s and do their own batching and kernel scheduling, so Python-side per-token
marking of them is not meaningful. Track-B latency is therefore measured with the native
tool (``vllm bench``, ``llama-bench``), parsed into ``schema.Latency``, and tagged with the
backend. vLLM tok/s and llama.cpp tok/s never share a latency column with each other or
with HF, exactly as an INT4 number never shares a column with an FP16 number on a
different backend.

The two parsers are pure and CPU-tested against captured JSON. The probes that invoke the
tools and read GPU memory require the pod and are gated; only their command construction
and assembly live here.

UNVALIDATED, POD GATE (see docs/decisions.md 2026-07-16): ``NativeVllmLatency`` is not yet
correct and MUST be fixed on the pod before any Track-B latency run. Two known faults, both
needing a real GPU to settle, so they are not guessed here:

- It benchmarks the wrong artifact. ``_vllm_bench_command`` points ``vllm bench`` at
  ``variant.model.model_id`` (the base FP16 model), while ``weights_disk_mb`` reads
  ``provider.model`` (the served compressed-tensors checkpoint). The bench must target the
  served ``checkpoint_path``, the same weights the eval scored.
- ``vllm bench serve`` needs a running vLLM server that nothing here starts. The pod fix
  decides ``bench serve`` (start a server, drive it) vs ``bench latency`` (offline), then
  matches ``parse_vllm_bench``'s expected JSON keys to whichever tool's output is used.

``NativeLlamaCppLatency`` and both pure parsers are not implicated; only the vLLM probe's
command construction and server wiring are unsettled.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from frontier.latency.machine import NvidiaSmiProbe, probe_clock_lock, to_machine_state
from frontier.latency.memory import kv_cache_dims, kv_cache_mb
from frontier.latency.rig import LatencyMemory, cost_proxy
from frontier.latency.stats import MEDIAN_Q, MS_PER_S, P95_Q, percentile
from frontier.schema import Latency, MachineState, Memory, RunMode

if TYPE_CHECKING:
    from frontier.eval.provider import LogitProvider
    from frontier.pipeline.config import ResolvedConfig

# The neutral no-GPU machine state a pure parser carries; the pod probe replaces it with a
# live nvidia-smi reading captured around the benchmark.
ABSENT_MACHINE_STATE = MachineState(0, 0, 0, 0.0, clocks_locked=False, clock_drift_flag=False)


def parse_vllm_bench(
    payload: Mapping[str, Any], *, batch_size: int, machine_state: MachineState | None = None
) -> Latency:
    """Parse a ``vllm bench serve --save-result`` JSON into one ``Latency``.

    vLLM reports TTFT and TPOT (time-per-output-token, the ITL clock) median and p95
    directly, plus ``output_throughput``. TTFT and ITL stay in separate columns, matching
    the two-clocks rule. Requires the p95 percentile keys, so the benchmark is run with
    ``--percentile-metrics ttft,tpot --metric-percentiles 95``.
    """
    return Latency(
        batch_size=batch_size,
        ttft_median_ms=float(payload["median_ttft_ms"]),
        ttft_p95_ms=float(payload["p95_ttft_ms"]),
        itl_median_ms=float(payload["median_tpot_ms"]),
        itl_p95_ms=float(payload["p95_tpot_ms"]),
        throughput_tok_s=float(payload["output_throughput"]),
        n_trials=int(payload["completed"]),
        warmup_discarded=0,
        machine_state=machine_state or ABSENT_MACHINE_STATE,
    )


def parse_llama_bench(
    runs: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    context_len: int,
    machine_state: MachineState | None = None,
) -> Latency:
    """Parse ``llama-bench -o json`` runs into one ``Latency``.

    llama-bench emits one entry per test with a per-repetition tokens/sec sample list.
    The prompt-processing (``pp``) entry is the prefill clock: each ``pp`` tok/s sample
    maps to ``ttft_ms = context_len / tok_s * 1000``. The text-generation (``tg``) entry
    is the decode clock: each ``tg`` tok/s sample maps to ``itl_ms = 1000 / tok_s``.
    Median and p95 come from those derived per-sample distributions; throughput is the
    median ``tg`` rate. ``warmup_discarded`` is 0 because llama-bench discards its own.
    """
    pp_samples = _samples(_find_run(runs, gen=False))
    tg_samples = _samples(_find_run(runs, gen=True))
    ttft = [context_len / rate * MS_PER_S for rate in pp_samples]
    itl = [MS_PER_S / rate for rate in tg_samples]
    return Latency(
        batch_size=batch_size,
        ttft_median_ms=percentile(ttft, MEDIAN_Q),
        ttft_p95_ms=percentile(ttft, P95_Q),
        itl_median_ms=percentile(itl, MEDIAN_Q),
        itl_p95_ms=percentile(itl, P95_Q),
        throughput_tok_s=percentile(tg_samples, MEDIAN_Q),
        n_trials=len(tg_samples),
        warmup_discarded=0,
        machine_state=machine_state or ABSENT_MACHINE_STATE,
    )


def _find_run(runs: Sequence[Mapping[str, Any]], *, gen: bool) -> Mapping[str, Any]:
    """The text-generation run (``gen=True``) or the prompt-processing run.

    A ``tg`` run generates tokens (``n_gen > 0`` and no prompt); a ``pp`` run only
    processes the prompt (``n_prompt > 0`` and no generation). Raises ``ValueError`` if the
    requested run is not among ``runs``.
    """
    for run in runs:
        n_prompt = int(run.get("n_prompt", 0))
        n_gen = int(run.get("n_gen", 0))
        if gen and n_gen > 0 and n_prompt == 0:
            return run
        if not gen and n_prompt > 0 and n_gen == 0:
            return run
    kind = "text-generation" if gen else "prompt-processing"
    raise ValueError(f"no {kind} run in llama-bench output")


def _samples(run: Mapping[str, Any]) -> list[float]:
    raw = run.get("samples_ts")
    if isinstance(raw, Sequence) and raw:
        return [float(sample) for sample in raw]
    return [float(run["avg_ts"])]


class NativeVllmLatency:
    """The vLLM ``LatencyProbe``: run ``vllm bench`` per batch size and parse the JSON.

    UNVALIDATED. See the module docstring and docs/decisions.md (2026-07-16): this probe
    benchmarks the base model instead of the served checkpoint and assumes a running
    ``vllm bench serve`` server that is not started here. It must be corrected on the pod
    against real vLLM before any Track-B latency number from it is trusted.
    """

    def __init__(self, run_json: Callable[[Sequence[str]], Any] | None = None) -> None:
        self._run_json = run_json or _run_vllm_bench

    def __call__(
        self,
        provider: LogitProvider,
        resolved: ResolvedConfig,
        *,
        device: str,  # noqa: ARG002
        mode: RunMode,  # noqa: ARG002
    ) -> LatencyMemory:  # pragma: no cover
        spec = resolved.variant.latency
        machine = NvidiaSmiProbe()
        locked = probe_clock_lock()
        latency: list[Latency] = []
        throughput: dict[int, float] = {}
        for batch_size in spec.batch_sizes:
            before = machine.capture()
            payload = self._run_json(_vllm_bench_command(resolved, batch_size))
            state = to_machine_state(before, machine.capture(), clocks_locked=locked)
            row = parse_vllm_bench(payload, batch_size=batch_size, machine_state=state)
            latency.append(row)
            throughput[batch_size] = row.throughput_tok_s
        weights_path = Path(str(cast(Any, provider).model))
        return _assemble(resolved, latency, throughput, weights_path=weights_path)


class NativeLlamaCppLatency:
    """The llama.cpp ``LatencyProbe``: run ``llama-bench`` per batch size and parse the JSON.

    Before any number is trusted it asserts full CUDA offload: the reported
    ``n_gpu_layers`` must cover every layer, or a partial offload has silently turned the
    decode clock into a CPU number.
    """

    def __init__(self, run_json: Callable[[Sequence[str]], Any] | None = None) -> None:
        self._run_json = run_json or _run_llama_bench

    def __call__(
        self,
        provider: LogitProvider,
        resolved: ResolvedConfig,
        *,
        device: str,  # noqa: ARG002
        mode: RunMode,  # noqa: ARG002
    ) -> LatencyMemory:  # pragma: no cover
        spec = resolved.variant.latency
        gguf_path = Path(str(cast(Any, provider).gguf_path))
        machine = NvidiaSmiProbe()
        locked = probe_clock_lock()
        context_len = spec.context_lengths[0]
        latency: list[Latency] = []
        throughput: dict[int, float] = {}
        for batch_size in spec.batch_sizes:
            before = machine.capture()
            runs = self._run_json(
                _llama_bench_command(gguf_path, resolved, batch_size, context_len)
            )
            assert_full_offload(runs, model_layers=_model_layers(resolved))
            state = to_machine_state(before, machine.capture(), clocks_locked=locked)
            row = parse_llama_bench(
                runs, batch_size=batch_size, context_len=context_len, machine_state=state
            )
            latency.append(row)
            throughput[batch_size] = row.throughput_tok_s
        return _assemble(resolved, latency, throughput, weights_path=gguf_path)


def assert_full_offload(runs: Sequence[Mapping[str, Any]], *, model_layers: int) -> None:
    """Raise unless every llama-bench run offloaded the whole model to the GPU.

    A GGUF decode number is only honest with zero CPU spill, so a run whose reported
    ``n_gpu_layers`` is below the model's layer count fails loudly rather than reporting a
    silently CPU-bound tok/s.
    """
    for run in runs:
        reported = int(run.get("n_gpu_layers", 0))
        if reported < model_layers:
            raise ValueError(
                f"llama-bench offloaded {reported} of {model_layers} layers; "
                f"a partial offload makes the decode clock a CPU number"
            )


def _assemble(
    resolved: ResolvedConfig,
    latency: list[Latency],
    throughput: Mapping[int, float],
    *,
    weights_path: Path,
) -> LatencyMemory:  # pragma: no cover
    spec = resolved.variant.latency
    disk_mb = _path_size_mb(weights_path)
    n_layers, n_kv_heads, head_dim = _model_dims(resolved)
    kv_bytes = 2
    memory: list[Memory] = []
    peak: dict[tuple[int, int], float] = {}
    for batch_size in spec.batch_sizes:
        for context_len in spec.context_lengths:
            vram = _nvidia_used_mb()
            kv = kv_cache_mb(
                n_layers=n_layers,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                seq_len=context_len,
                batch_size=batch_size,
                dtype_bytes=kv_bytes,
            )
            memory.append(
                Memory(
                    batch_size=batch_size,
                    context_len=context_len,
                    peak_vram_mb=vram,
                    weights_disk_mb=disk_mb,
                    weights_resident_mb=disk_mb,
                    kv_cache_mb=kv,
                )
            )
            peak[(batch_size, context_len)] = vram
    reference = max(spec.batch_sizes)
    tok_s_per_gb = cost_proxy(throughput[reference], peak[(reference, spec.context_lengths[0])])
    return LatencyMemory(latency=latency, memory=memory, tok_s_per_gb=tok_s_per_gb)


def _vllm_bench_command(resolved: ResolvedConfig, batch_size: int) -> list[str]:  # pragma: no cover
    return [
        "vllm",
        "bench",
        "serve",
        "--model",
        resolved.variant.model.model_id,
        "--max-concurrency",
        str(batch_size),
        "--percentile-metrics",
        "ttft,tpot",
        "--metric-percentiles",
        "95",
        "--save-result",
    ]


def _llama_bench_command(
    gguf_path: Path, resolved: ResolvedConfig, batch_size: int, context_len: int
) -> list[str]:  # pragma: no cover
    layers = resolved.backend["gpu_offload_layers"]
    return [
        "llama-bench",
        "-m",
        str(gguf_path),
        "-ngl",
        str(layers),
        "-b",
        str(batch_size),
        "-p",
        str(context_len),
        "-n",
        "128",
        "-o",
        "json",
    ]


def _run_vllm_bench(command: Sequence[str]) -> Any:  # pragma: no cover
    subprocess.run(list(command), check=True)
    with Path("benchmark_serve.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_llama_bench(command: Sequence[str]) -> Any:  # pragma: no cover
    result = subprocess.run(list(command), check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _model_dims(resolved: ResolvedConfig) -> tuple[int, int, int]:  # pragma: no cover
    import transformers  # noqa: PLC0415

    config = transformers.AutoConfig.from_pretrained(resolved.variant.model.model_id)
    return kv_cache_dims(config)


def _model_layers(resolved: ResolvedConfig) -> int:  # pragma: no cover
    return _model_dims(resolved)[0]


def _path_size_mb(path: Path) -> float:  # pragma: no cover
    if path.is_file():
        return path.stat().st_size / 1e6
    total = sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return total / 1e6


def _nvidia_used_mb() -> float:  # pragma: no cover
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip().splitlines()[0])
