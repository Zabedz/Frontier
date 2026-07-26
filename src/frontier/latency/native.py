"""Track-B latency from each serving stack's own benchmarker.

The WP4 CUDA-event rig drives an ``nn.Module`` token by token; vLLM and llama.cpp are not
``nn.Module``s and do their own batching and kernel scheduling, so Python-side per-token
marking of them is not meaningful. Track-B latency is therefore measured with the native
tool (``vllm bench serve``, ``llama-bench``), parsed into ``schema.Latency``, and tagged
with the backend. vLLM tok/s and llama.cpp tok/s never share a latency column with each
other or with HF, exactly as an INT4 number never shares a column with an FP16 number on
a different backend.

The vLLM probe benches the artifact the eval scored. It serves ``provider.model`` (the
compressed-tensors checkpoint dir, or the base model id for the FP16 fidelity gate)
itself, drives ``vllm bench serve`` against that server once per batch size, and tears
the server down when done. The runner measures latency before the eval touches the
provider, and the provider builds its engine lazily, so the server has the GPU to
itself. ``bench serve`` is the tool because its result JSON carries the median/p95 TTFT
and TPOT split ``schema.Latency`` requires; ``bench latency`` reports a single
end-to-end time, which cannot fill the two clocks. Command construction and the result
keys are written against the vllm 0.25.1 source (``vllm/benchmarks/serve.py``: the
``--percentile-metrics ttft,tpot --metric-percentiles 95`` run emits exactly the keys
``parse_vllm_bench`` reads, and the bench waits up to 600s for the endpoint, absorbing
the server's model-load window). The flow is validated once, watched, on the pod before
any Track-B row is banked (docs/decisions.md 2026-07-17).

Peak VRAM is read while the workload is alive: a point read under the still-running
server for vLLM, a background ``VramSampler`` around the ``llama-bench`` process for
GGUF. An ``nvidia-smi`` read taken after the process exits sees an idle GPU and would
inflate ``tok_s_per_gb`` by dividing throughput by a few idle megabytes.

The two parsers are pure and CPU-tested against captured JSON. The subprocess runners,
the server start, and the nvidia-smi reads are injectable, so the probes' control flow
is CPU-tested too; only the real tool invocations need the pod.
"""

from __future__ import annotations

import json
import socket
import subprocess
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from frontier.latency.machine import (
    MachineProbe,
    NvidiaSmiProbe,
    probe_clock_lock,
    to_machine_state,
)
from frontier.latency.memory import kv_cache_dims, kv_cache_mb
from frontier.latency.rig import FULL_DECODE_LEN, LatencyMemory, cost_proxy
from frontier.latency.stats import MEDIAN_Q, MS_PER_S, P95_Q, percentile
from frontier.schema import Latency, MachineState, Memory, RunMode

if TYPE_CHECKING:
    from frontier.eval.provider import LogitProvider
    from frontier.pipeline.config import ResolvedConfig

# The neutral no-GPU machine state a pure parser carries; the pod probe replaces it with a
# live nvidia-smi reading captured around the benchmark.
ABSENT_MACHINE_STATE = MachineState(0, 0, 0, 0.0, clocks_locked=False, clock_drift_flag=False)

SERVER_STOP_TIMEOUT_S = 30.0
VRAM_SAMPLE_INTERVAL_S = 0.5
LOG_TAIL_LINES = 40
# llama-bench generates single-stream, so a GGUF row carries one latency entry and labels
# it batch size 1 rather than repeating the same measurement under the eval's batch axis.
SINGLE_STREAM_BATCH = 1
LLAMA_BENCH_DECODE_LEN = 128
# The same fraction the eval engine requests (backends/vllm.py), so the bench's resident
# pool is the pool the eval ran under; it is also the peak-VRAM definition for vLLM rows
# (docs/methodology.md): vLLM preallocates, so peak is the pool, and this pins it against
# an upstream default change.
SERVE_GPU_MEMORY_UTILIZATION = 0.9


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


class VramSampler:
    """Peak GPU memory over a workload's lifetime, sampled on a background thread.

    nvidia-smi reports the moment's usage, so a reading taken after a benchmark process
    exits sees an idle GPU. The sampler reads once on entry, every ``interval_s`` while
    the workload runs, and once on exit; ``peak_mb`` is the max seen. Both the loop and
    the exit read only ever raise ``peak_mb``, so the unsynchronised max is safe.
    """

    def __init__(
        self, read_mb: Callable[[], float], interval_s: float = VRAM_SAMPLE_INTERVAL_S
    ) -> None:
        self._read_mb = read_mb
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mb = 0.0

    def __enter__(self) -> VramSampler:
        self.peak_mb = self._read_mb()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_mb = max(self.peak_mb, self._read_mb())

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.peak_mb = max(self.peak_mb, self._read_mb())


class VllmServer:
    """A ``vllm serve`` process the probe owns: poll-guarded, stopped in ``finally``.

    The server's stdout/stderr stream to ``log_path``: vLLM logs every request, and an
    undrained PIPE would fill and deadlock the server mid-bench. The log tail rides along
    in the failure message when the server dies, because the interesting error (an OOM, a
    bad checkpoint) is in the server's output, and the bench client only sees a
    connection refusal.
    """

    def __init__(self, process: Any, log_path: Path) -> None:
        self.process = process
        self.log_path = log_path

    def assert_alive(self) -> None:
        """Raise ``RuntimeError`` with the log tail if the server process has exited."""
        code = self.process.poll()
        if code is not None:
            raise RuntimeError(
                f"vllm serve exited with code {code} during the latency bench; "
                f"server log tail:\n{self.log_tail()}"
            )

    def log_tail(self, lines: int = LOG_TAIL_LINES) -> str:
        if not self.log_path.exists():
            return "<no server log>"
        return "\n".join(self.log_path.read_text(encoding="utf-8").splitlines()[-lines:])

    def stop(self) -> None:
        """Terminate the server, escalating to SIGKILL after ``SERVER_STOP_TIMEOUT_S``."""
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=SERVER_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


class NativeVllmLatency:
    """The vLLM ``LatencyProbe``: serve the eval's artifact, ``vllm bench`` it per batch.

    The served model is ``provider.model``, the same weights the eval scored and
    ``weights_disk_mb`` measures: the compressed-tensors checkpoint dir for a quantised
    variant, the base model id for the FP16 fidelity gate. The probe owns the server
    lifecycle; ``vllm bench serve`` performs its own endpoint ready-wait, and
    ``assert_alive`` fails loudly, with the server log tail, when the server has died;
    the bench would otherwise sit out its full ready-wait against a dead port. Every
    collaborator that would touch a GPU or a subprocess is injectable, so the whole
    control flow runs under CPU tests.
    """

    def __init__(
        self,
        run_json: Callable[[Sequence[str], Path], Any] | None = None,
        *,
        server_factory: Callable[..., VllmServer] | None = None,
        pick_port: Callable[[], int] | None = None,
        read_vram_mb: Callable[[], float] | None = None,
        model_dims: Callable[[ResolvedConfig], tuple[int, int, int]] | None = None,
        machine_probe: MachineProbe | None = None,
        clock_lock: Callable[[], bool] | None = None,
    ) -> None:
        self._run_json = run_json or _run_vllm_bench
        self._server_factory = server_factory or _start_vllm_server
        self._pick_port = pick_port or _free_port
        self._read_vram_mb = read_vram_mb or _nvidia_used_mb
        self._model_dims = model_dims or _model_dims
        self._machine = machine_probe or NvidiaSmiProbe()
        self._clock_lock = clock_lock or probe_clock_lock

    def __call__(
        self,
        provider: LogitProvider,
        resolved: ResolvedConfig,
        *,
        device: str,  # noqa: ARG002
        mode: RunMode,  # noqa: ARG002
    ) -> LatencyMemory:
        spec = resolved.variant.latency
        model = str(cast(Any, provider).model)
        machine = self._machine
        locked = self._clock_lock()
        latency: list[Latency] = []
        throughput: dict[int, float] = {}
        vram_by_batch: dict[int, float] = {}
        port = self._pick_port()
        with tempfile.TemporaryDirectory(prefix="vllm-bench-") as tmp:
            tmp_path = Path(tmp)
            server = self._server_factory(model, port=port, log_path=tmp_path / "serve.log")
            try:
                for batch_size in spec.batch_sizes:
                    server.assert_alive()
                    before = machine.capture()
                    result_path = tmp_path / f"bench_b{batch_size}.json"
                    command = _vllm_bench_command(
                        model, resolved, batch_size, port=port, result_path=result_path
                    )
                    try:
                        payload = self._run_json(command, result_path)
                    except subprocess.CalledProcessError as error:
                        raise RuntimeError(
                            f"vllm bench serve failed for batch {batch_size} "
                            f"(exit {error.returncode}); server log tail:\n{server.log_tail()}"
                        ) from error
                    server.assert_alive()
                    vram_by_batch[batch_size] = self._read_vram_mb()
                    state = to_machine_state(before, machine.capture(), clocks_locked=locked)
                    row = parse_vllm_bench(payload, batch_size=batch_size, machine_state=state)
                    latency.append(row)
                    throughput[batch_size] = row.throughput_tok_s
            finally:
                server.stop()
        return _assemble(
            resolved,
            latency,
            throughput,
            weights_path=Path(model),
            vram_by_batch=vram_by_batch,
            dims=self._model_dims(resolved),
        )


class NativeLlamaCppLatency:
    """The llama.cpp ``LatencyProbe``: run ``llama-bench`` per batch size and parse the JSON.

    Before any number is trusted it asserts full CUDA offload: the reported
    ``n_gpu_layers`` must cover every layer, or a partial offload has silently turned the
    decode clock into a CPU number. Peak VRAM comes from a ``VramSampler`` running while
    llama-bench does, because the process (and its memory) is gone by parse time.
    """

    def __init__(
        self,
        run_json: Callable[[Sequence[str]], Any] | None = None,
        *,
        read_vram_mb: Callable[[], float] | None = None,
        model_dims: Callable[[ResolvedConfig], tuple[int, int, int]] | None = None,
        machine_probe: MachineProbe | None = None,
        clock_lock: Callable[[], bool] | None = None,
    ) -> None:
        self._run_json = run_json or _run_llama_bench
        self._read_vram_mb = read_vram_mb or _nvidia_used_mb
        self._model_dims = model_dims or _model_dims
        self._machine = machine_probe or NvidiaSmiProbe()
        self._clock_lock = clock_lock or probe_clock_lock

    def __call__(
        self,
        provider: LogitProvider,
        resolved: ResolvedConfig,
        *,
        device: str,  # noqa: ARG002
        mode: RunMode,  # noqa: ARG002
    ) -> LatencyMemory:
        spec = resolved.variant.latency
        gguf_path = Path(str(cast(Any, provider).gguf_path))
        dims = self._model_dims(resolved)
        machine = self._machine
        locked = self._clock_lock()
        context_len = spec.context_lengths[0]
        before = machine.capture()
        with VramSampler(self._read_vram_mb) as sampler:
            runs = self._run_json(_llama_bench_command(gguf_path, resolved, context_len))
        assert_full_offload(runs, model_layers=dims[0])
        state = to_machine_state(before, machine.capture(), clocks_locked=locked)
        row = parse_llama_bench(
            runs, batch_size=SINGLE_STREAM_BATCH, context_len=context_len, machine_state=state
        )
        return _assemble(
            resolved,
            [row],
            {SINGLE_STREAM_BATCH: row.throughput_tok_s},
            weights_path=gguf_path,
            vram_by_batch={SINGLE_STREAM_BATCH: sampler.peak_mb},
            dims=dims,
            batch_sizes=(SINGLE_STREAM_BATCH,),
        )


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
    vram_by_batch: Mapping[int, float],
    dims: tuple[int, int, int],
    batch_sizes: Sequence[int] | None = None,
) -> LatencyMemory:
    spec = resolved.variant.latency
    # A single-stream backend measures one batch size and says so, rather than reporting
    # the eval's batch axis it never exercised.
    sizes = tuple(batch_sizes) if batch_sizes is not None else spec.batch_sizes
    disk_mb = _path_size_mb(weights_path)
    n_layers, n_kv_heads, head_dim = dims
    kv_bytes = 2
    memory: list[Memory] = []
    for batch_size in sizes:
        for context_len in spec.context_lengths:
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
                    peak_vram_mb=vram_by_batch[batch_size],
                    weights_disk_mb=disk_mb,
                    weights_resident_mb=disk_mb,
                    kv_cache_mb=kv,
                )
            )
    reference = max(sizes)
    tok_s_per_gb = cost_proxy(throughput[reference], vram_by_batch[reference])
    return LatencyMemory(latency=latency, memory=memory, tok_s_per_gb=tok_s_per_gb)


def _vllm_serve_command(model: str, *, port: int) -> list[str]:
    """The ``vllm serve`` invocation the probe owns.

    Prefix caching is off so every bench run pays a cold prefill: the three batch sizes
    reuse one server, and the random dataset can repeat prompt prefixes across runs,
    which would let a cached prefill warm the later TTFT readings. The memory fraction is
    pinned to the eval engine's value so the resident pool, which is what the vLLM peak
    VRAM read reports, is the pool the eval ran under.
    """
    return [
        "vllm",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        str(SERVE_GPU_MEMORY_UTILIZATION),
    ]


def _vllm_bench_command(
    model: str, resolved: ResolvedConfig, batch_size: int, *, port: int, result_path: Path
) -> list[str]:
    """The ``vllm bench serve`` invocation for one batch size.

    ``--num-prompts`` is ``batch_size * n_trials`` so every concurrency slot sees
    ``n_trials`` requests. Input and output lengths mirror the WP4 rig's operating point
    (fixed prefill at ``context_lengths[0]``, ``FULL_DECODE_LEN`` decode), so the HF and
    vLLM clocks are read under the same load shape even though their columns never mix;
    ``--ignore-eos`` holds the decode length there, since a model may emit EOS early on
    random-token prompts and each variant would drift off the shape by its own amount.
    """
    spec = resolved.variant.latency
    return [
        "vllm",
        "bench",
        "serve",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(spec.context_lengths[0]),
        "--random-output-len",
        str(FULL_DECODE_LEN),
        "--num-prompts",
        str(batch_size * spec.n_trials),
        "--max-concurrency",
        str(batch_size),
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot",
        "--metric-percentiles",
        "95",
        "--save-result",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
    ]


def _llama_bench_command(gguf_path: Path, resolved: ResolvedConfig, context_len: int) -> list[str]:
    """The ``llama-bench`` invocation for one GGUF measurement.

    ``-b`` is deliberately not set. It is llama.cpp's prefill chunk size, not a count of
    concurrent requests, so feeding it the eval's batch sizes measured how fast a prompt
    is chunked rather than how the model serves load: ``-b 1`` prefills one token at a
    time and reports a TTFT an order of magnitude off a real deployment's, while the
    decode rate does not move at all because llama-bench generates single-stream. The
    default chunk size is what a llama.cpp deployment runs. ``-r`` carries the spec's
    trial count so the GGUF track meets the same repetition floor as the HF rig.
    """
    layers = resolved.backend["gpu_offload_layers"]
    return [
        "llama-bench",
        "-m",
        str(gguf_path),
        "-ngl",
        str(layers),
        "-p",
        str(context_len),
        "-n",
        str(LLAMA_BENCH_DECODE_LEN),
        "-r",
        str(resolved.variant.latency.n_trials),
        "-o",
        "json",
    ]


def _start_vllm_server(model: str, *, port: int, log_path: Path) -> VllmServer:  # pragma: no cover
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            _vllm_serve_command(model, port=port), stdout=handle, stderr=subprocess.STDOUT
        )
    return VllmServer(process, log_path)


def _free_port() -> int:  # pragma: no cover
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_vllm_bench(command: Sequence[str], result_path: Path) -> Any:  # pragma: no cover
    subprocess.run(list(command), check=True)
    with result_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_llama_bench(command: Sequence[str]) -> Any:  # pragma: no cover
    result = subprocess.run(list(command), check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _model_dims(resolved: ResolvedConfig) -> tuple[int, int, int]:  # pragma: no cover
    import transformers  # noqa: PLC0415

    config = transformers.AutoConfig.from_pretrained(resolved.variant.model.model_id)
    return kv_cache_dims(config)


def _path_size_mb(path: Path) -> float:
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
