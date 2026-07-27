"""The native benchmarkers: JSON parsers, probe control flow, server lifecycle, VRAM."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from frontier.eval.provider import LogitProvider
from frontier.latency.machine import ClockReading
from frontier.latency.native import (
    ABSENT_MACHINE_STATE,
    MAX_GPU_MEMORY_FRACTION,
    SINGLE_STREAM_BATCH,
    VLLM_MEMORY_BUDGET_MB,
    NativeLlamaCppLatency,
    NativeVllmLatency,
    VllmServer,
    VramSampler,
    _vllm_bench_command,
    _vllm_serve_command,
    assert_full_offload,
    parse_llama_bench,
    parse_vllm_bench,
    vllm_memory_fraction,
)
from frontier.latency.rig import FULL_DECODE_LEN, cost_proxy
from frontier.latency.stats import MEDIAN_Q, MS_PER_S, P95_Q, percentile
from frontier.pipeline.config import ResolvedConfig, resolve_config
from frontier.schema import MachineState

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"

VLLM_PAYLOAD = {
    "completed": 100,
    "median_ttft_ms": 42.0,
    "p95_ttft_ms": 88.0,
    "median_tpot_ms": 12.0,
    "p95_tpot_ms": 20.0,
    "output_throughput": 830.0,
    "request_throughput": 6.5,
}

PP_SAMPLES = [1000.0, 1010.0, 990.0, 1005.0]
TG_SAMPLES = [50.0, 52.0, 48.0, 51.0]
CONTEXT_LEN = 512
BATCH = 4
MODEL_LAYERS = 36
DIMS = (MODEL_LAYERS, 8, 128)
SERVED_VRAM_MB = 9000.0
SERVED_GPU_FRACTION = 0.5699
SAMPLED_VRAM_MB = 7000.0
PEAK_READ_MB = 4000.0


def _llama_runs() -> list[dict[str, object]]:
    return [
        {"n_prompt": CONTEXT_LEN, "n_gen": 0, "n_gpu_layers": 99, "samples_ts": PP_SAMPLES},
        {"n_prompt": 0, "n_gen": 128, "n_gpu_layers": 99, "samples_ts": TG_SAMPLES},
    ]


def _resolved(name: str) -> ResolvedConfig:
    return resolve_config(CONFIG_ROOT / "variants" / f"{name}.yaml", config_root=CONFIG_ROOT)


def _value_after(command: Sequence[str], flag: str) -> str:
    return command[list(command).index(flag) + 1]


class _VllmProvider:
    def __init__(self, model: str) -> None:
        self.model = model


class _GgufProvider:
    def __init__(self, gguf_path: str) -> None:
        self.gguf_path = gguf_path


class _FakeMachine:
    """A machine probe with a fixed live reading; keeps nvidia-smi out of the tests."""

    def __init__(self, sm_mhz: int = 1800) -> None:
        self.reading = ClockReading(sm_mhz, 9001, 55, 120.0, present=True)

    def capture(self) -> ClockReading:
        return self.reading


class _FakeProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True
        self.exit_code = 0

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
        return self.exit_code if self.exit_code is not None else 0

    def kill(self) -> None:
        self.exit_code = -9


def test_parse_vllm_bench_maps_two_clocks_and_throughput() -> None:
    latency = parse_vllm_bench(VLLM_PAYLOAD, batch_size=BATCH)
    assert latency.batch_size == BATCH
    assert latency.ttft_median_ms == VLLM_PAYLOAD["median_ttft_ms"]
    assert latency.ttft_p95_ms == VLLM_PAYLOAD["p95_ttft_ms"]
    assert latency.itl_median_ms == VLLM_PAYLOAD["median_tpot_ms"]
    assert latency.itl_p95_ms == VLLM_PAYLOAD["p95_tpot_ms"]
    assert latency.throughput_tok_s == VLLM_PAYLOAD["output_throughput"]
    assert latency.n_trials == VLLM_PAYLOAD["completed"]
    assert latency.warmup_discarded == 0
    assert latency.machine_state is ABSENT_MACHINE_STATE


def test_parse_vllm_bench_takes_an_injected_machine_state() -> None:
    state = MachineState(1800, 9000, 55, 120.0, clocks_locked=False, clock_drift_flag=False)
    latency = parse_vllm_bench(VLLM_PAYLOAD, batch_size=1, machine_state=state)
    assert latency.machine_state is state


def test_parse_llama_bench_converts_prefill_and_decode_clocks() -> None:
    latency = parse_llama_bench(_llama_runs(), batch_size=1, context_len=CONTEXT_LEN)
    ttft = [CONTEXT_LEN / rate * MS_PER_S for rate in PP_SAMPLES]
    itl = [MS_PER_S / rate for rate in TG_SAMPLES]
    assert latency.ttft_median_ms == pytest.approx(percentile(ttft, MEDIAN_Q))
    assert latency.ttft_p95_ms == pytest.approx(percentile(ttft, P95_Q))
    assert latency.itl_median_ms == pytest.approx(percentile(itl, MEDIAN_Q))
    assert latency.itl_p95_ms == pytest.approx(percentile(itl, P95_Q))
    assert latency.throughput_tok_s == pytest.approx(percentile(TG_SAMPLES, MEDIAN_Q))
    assert latency.n_trials == len(TG_SAMPLES)
    assert latency.warmup_discarded == 0


def test_parse_llama_bench_falls_back_to_avg_ts() -> None:
    runs = [
        {"n_prompt": CONTEXT_LEN, "n_gen": 0, "n_gpu_layers": 99, "avg_ts": 1000.0},
        {"n_prompt": 0, "n_gen": 128, "n_gpu_layers": 99, "avg_ts": 50.0},
    ]
    latency = parse_llama_bench(runs, batch_size=1, context_len=CONTEXT_LEN)
    assert latency.itl_median_ms == pytest.approx(MS_PER_S / 50.0)
    assert latency.ttft_median_ms == pytest.approx(CONTEXT_LEN / 1000.0 * MS_PER_S)
    assert latency.n_trials == 1


def test_parse_llama_bench_requires_both_runs() -> None:
    only_pp = [{"n_prompt": CONTEXT_LEN, "n_gen": 0, "n_gpu_layers": 99, "avg_ts": 1000.0}]
    with pytest.raises(ValueError, match="text-generation run"):
        parse_llama_bench(only_pp, batch_size=1, context_len=CONTEXT_LEN)


def test_assert_full_offload_passes_and_fails() -> None:
    assert_full_offload(_llama_runs(), model_layers=MODEL_LAYERS)
    partial = [{"n_prompt": 0, "n_gen": 128, "n_gpu_layers": MODEL_LAYERS - 1}]
    with pytest.raises(ValueError, match="partial offload"):
        assert_full_offload(partial, model_layers=MODEL_LAYERS)


def test_vllm_serve_command_targets_model_and_port() -> None:
    command = _vllm_serve_command("/ckpt/int4-gptq", port=9000, gpu_fraction=0.5699)
    assert command[:3] == ["vllm", "serve", "/ckpt/int4-gptq"]
    assert _value_after(command, "--port") == "9000"
    assert _value_after(command, "--host") == "127.0.0.1"
    assert "--no-enable-prefix-caching" in command
    assert _value_after(command, "--gpu-memory-utilization") == "0.5699"


@pytest.mark.parametrize("total_mb", [16380.0, 20475.0, 24564.0, 49140.0])
def test_vllm_memory_fraction_reserves_the_same_budget_on_every_card(total_mb: float) -> None:
    reserved = vllm_memory_fraction(total_mb) * total_mb
    assert reserved == pytest.approx(VLLM_MEMORY_BUDGET_MB, rel=1e-3)


def test_vllm_memory_fraction_rejects_a_card_too_small_for_the_budget() -> None:
    too_small = VLLM_MEMORY_BUDGET_MB / MAX_GPU_MEMORY_FRACTION - 1.0
    with pytest.raises(ValueError, match="above the"):
        vllm_memory_fraction(too_small)


def test_vllm_bench_command_matches_parser_and_rig_operating_point() -> None:
    resolved = _resolved("int4-gptq")
    spec = resolved.variant.latency
    result_path = Path("/scratch/bench_b4.json")
    command = _vllm_bench_command(
        "/ckpt/int4-gptq", resolved, BATCH, port=8123, result_path=result_path
    )
    assert command[:3] == ["vllm", "bench", "serve"]
    assert _value_after(command, "--model") == "/ckpt/int4-gptq"
    assert _value_after(command, "--base-url") == "http://127.0.0.1:8123"
    assert _value_after(command, "--dataset-name") == "random"
    assert _value_after(command, "--random-input-len") == str(spec.context_lengths[0])
    assert _value_after(command, "--random-output-len") == str(FULL_DECODE_LEN)
    assert _value_after(command, "--num-prompts") == str(BATCH * spec.n_trials)
    assert _value_after(command, "--max-concurrency") == str(BATCH)
    assert _value_after(command, "--percentile-metrics") == "ttft,tpot"
    assert _value_after(command, "--metric-percentiles") == "95"
    assert _value_after(command, "--result-dir") == str(result_path.parent)
    assert _value_after(command, "--result-filename") == result_path.name
    assert "--save-result" in command
    assert "--ignore-eos" in command


def test_native_vllm_latency_serves_and_benches_the_eval_artifact(tmp_path: Path) -> None:
    resolved = _resolved("int4-gptq")
    spec = resolved.variant.latency
    checkpoint = tmp_path / "ckpt-int4-gptq"
    checkpoint.mkdir()
    (checkpoint / "weights.safetensors").write_bytes(b"x" * 2_000_000)
    process = _FakeProcess()
    served: list[tuple[str, int, float]] = []

    def factory(model: str, *, port: int, log_path: Path, gpu_fraction: float) -> VllmServer:
        served.append((model, port, gpu_fraction))
        return VllmServer(process, log_path)

    commands: list[list[str]] = []

    def run_json(command: Sequence[str], result_path: Path) -> dict[str, Any]:  # noqa: ARG001
        commands.append(list(command))
        return dict(VLLM_PAYLOAD)

    machine = _FakeMachine()
    probe = NativeVllmLatency(
        run_json,
        server_factory=factory,
        pick_port=lambda: 8123,
        read_vram_mb=lambda: SERVED_VRAM_MB,
        model_dims=lambda _: DIMS,
        machine_probe=machine,
        clock_lock=lambda: False,
        gpu_fraction=lambda: SERVED_GPU_FRACTION,
    )
    provider = cast(LogitProvider, _VllmProvider(str(checkpoint)))
    out = probe(provider, resolved, device="cuda", mode="full")

    assert served == [(str(checkpoint), 8123, SERVED_GPU_FRACTION)]
    assert len(out.latency) == len(spec.batch_sizes)
    state = out.latency[0].machine_state
    assert state.gpu_clock_sm_mhz == machine.reading.sm_mhz
    assert state.clocks_locked is False
    for command, batch_size in zip(commands, spec.batch_sizes, strict=True):
        assert _value_after(command, "--model") == str(checkpoint)
        assert resolved.variant.model.model_id not in command
        assert _value_after(command, "--max-concurrency") == str(batch_size)
        assert _value_after(command, "--num-prompts") == str(batch_size * spec.n_trials)
    assert all(entry.peak_vram_mb == SERVED_VRAM_MB for entry in out.memory)
    assert all(entry.weights_disk_mb == pytest.approx(2.0) for entry in out.memory)
    expected = cost_proxy(float(VLLM_PAYLOAD["output_throughput"]), SERVED_VRAM_MB)
    assert out.tok_s_per_gb == pytest.approx(expected)
    assert process.terminated


def test_native_vllm_latency_wraps_bench_failure_with_server_log() -> None:
    resolved = _resolved("int4-gptq")
    process = _FakeProcess()

    def factory(
        model: str,  # noqa: ARG001
        *,
        port: int,  # noqa: ARG001
        log_path: Path,
        gpu_fraction: float,  # noqa: ARG001
    ) -> VllmServer:
        log_path.write_text("engine crash: out of device memory\n", encoding="utf-8")
        return VllmServer(process, log_path)

    def run_json(command: Sequence[str], result_path: Path) -> dict[str, Any]:  # noqa: ARG001
        raise subprocess.CalledProcessError(returncode=1, cmd=list(command))

    probe = NativeVllmLatency(
        run_json,
        server_factory=factory,
        pick_port=lambda: 1,
        read_vram_mb=lambda: 0.0,
        model_dims=lambda _: DIMS,
        machine_probe=_FakeMachine(),
        clock_lock=lambda: False,
        gpu_fraction=lambda: SERVED_GPU_FRACTION,
    )
    provider = cast(LogitProvider, _VllmProvider("/ckpt"))
    with pytest.raises(RuntimeError, match="out of device memory"):
        probe(provider, resolved, device="cuda", mode="full")
    assert process.terminated


def test_vllm_server_assert_alive_raises_with_log_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "serve.log"
    log_path.write_text("loading weights\nCUDA out of memory\n", encoding="utf-8")
    process = _FakeProcess()
    process.exit_code = 3
    server = VllmServer(process, log_path)
    with pytest.raises(RuntimeError, match=r"exited with code 3(.|\n)*CUDA out of memory"):
        server.assert_alive()


def test_vllm_server_stop_escalates_to_kill() -> None:
    class _StuckProcess:
        def __init__(self) -> None:
            self.killed = False
            self._code: int | None = None

        def poll(self) -> int | None:
            return self._code

        def terminate(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None and not self.killed:
                raise subprocess.TimeoutExpired(cmd="vllm", timeout=timeout)
            self._code = -9
            return -9

        def kill(self) -> None:
            self.killed = True

    stuck = _StuckProcess()
    server = VllmServer(stuck, Path("/nonexistent-log"))
    server.stop()
    assert stuck.killed


def test_vram_sampler_reads_on_enter_and_exit() -> None:
    reads = [1000.0, PEAK_READ_MB]

    def reader() -> float:
        return reads.pop(0) if reads else 2000.0

    with VramSampler(reader, interval_s=60.0) as sampler:
        pass
    assert sampler.peak_mb == PEAK_READ_MB


def test_native_llama_latency_samples_vram_while_bench_runs(tmp_path: Path) -> None:
    resolved = _resolved("gguf-q4_k_m")
    spec = resolved.variant.latency
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"g" * 1_000_000)
    commands: list[list[str]] = []

    def run_json(command: Sequence[str]) -> list[dict[str, object]]:
        commands.append(list(command))
        return _llama_runs()

    probe = NativeLlamaCppLatency(
        run_json,
        read_vram_mb=lambda: SAMPLED_VRAM_MB,
        model_dims=lambda _: DIMS,
        machine_probe=_FakeMachine(),
        clock_lock=lambda: False,
    )
    provider = cast(LogitProvider, _GgufProvider(str(gguf)))
    out = probe(provider, resolved, device="cuda", mode="full")

    # llama-bench generates single-stream, so one measurement is taken and labelled batch
    # size 1 rather than repeated once per eval batch size.
    assert len(out.latency) == 1
    assert out.latency[0].batch_size == SINGLE_STREAM_BATCH
    assert len(commands) == 1
    assert all(entry.batch_size == SINGLE_STREAM_BATCH for entry in out.memory)
    assert len(out.memory) == len(spec.context_lengths)
    assert all(entry.peak_vram_mb == SAMPLED_VRAM_MB for entry in out.memory)
    assert all(entry.weights_disk_mb == pytest.approx(1.0) for entry in out.memory)
    expected = cost_proxy(percentile(TG_SAMPLES, MEDIAN_Q), SAMPLED_VRAM_MB)
    assert out.tok_s_per_gb == pytest.approx(expected)
    assert _value_after(commands[0], "-m") == str(gguf)
    # -b is llama.cpp's prefill chunk size, not concurrency, so the eval's batch axis must
    # never reach it; -r carries the spec's trial floor.
    assert "-b" not in commands[0]
    assert _value_after(commands[0], "-r") == str(spec.n_trials)


def test_native_llama_latency_rejects_partial_offload(tmp_path: Path) -> None:
    resolved = _resolved("gguf-q4_k_m")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"g")
    probe = NativeLlamaCppLatency(
        lambda _: _llama_runs(),
        read_vram_mb=lambda: SAMPLED_VRAM_MB,
        model_dims=lambda _: (100, 8, 128),
        machine_probe=_FakeMachine(),
        clock_lock=lambda: False,
    )
    provider = cast(LogitProvider, _GgufProvider(str(gguf)))
    with pytest.raises(ValueError, match="partial offload"):
        probe(provider, resolved, device="cuda", mode="full")
