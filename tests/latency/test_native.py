"""The native-benchmark parsers: vLLM bench and llama-bench JSON to schema.Latency."""

from __future__ import annotations

import pytest

from frontier.latency.native import (
    ABSENT_MACHINE_STATE,
    assert_full_offload,
    parse_llama_bench,
    parse_vllm_bench,
)
from frontier.latency.stats import MEDIAN_Q, MS_PER_S, P95_Q, percentile
from frontier.schema import MachineState

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


def _llama_runs() -> list[dict[str, object]]:
    return [
        {"n_prompt": CONTEXT_LEN, "n_gen": 0, "n_gpu_layers": 99, "samples_ts": PP_SAMPLES},
        {"n_prompt": 0, "n_gen": 128, "n_gpu_layers": 99, "samples_ts": TG_SAMPLES},
    ]


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
