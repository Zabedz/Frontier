"""CUDA-event timing and a short HF decode on real hardware. Skipped on every CPU box.

The CPU suite exercises the wall-clock path and the stubbed event bookkeeping; this is
the only place the real ``CudaEventClock`` and a real decode loop run, so it is gated on
a live CUDA device (and the model test also on ``FRONTIER_LIVE_MODELS``).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

import torch

from frontier.backends.hf import HFLogitProvider
from frontier.latency.timing import CudaEventClock, HFGenerationDriver, collect_trials

N_MARKS = 3
N_TOTAL = 3
DECODE_LEN = 8

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]


def test_cuda_event_clock_yields_positive_spans() -> None:
    clock = CudaEventClock(torch)
    matrix = torch.ones(512, 512, device="cuda")
    for _ in range(N_MARKS):
        (matrix @ matrix).sum()
        clock.mark()
    spans = clock.resolve()
    assert len(spans) == N_MARKS - 1
    assert all(span >= 0.0 for span in spans)


@pytest.mark.skipif(
    not os.environ.get("FRONTIER_LIVE_MODELS"),
    reason="live model download; set FRONTIER_LIVE_MODELS=1 to run",
)
def test_hf_driver_separates_ttft_from_itl_on_gpu() -> None:
    provider = HFLogitProvider(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct", device="cuda", weight_dtype="fp16"
    )
    model, tokenizer = provider.loaded_model()
    driver = HFGenerationDriver(model, tokenizer, "cuda")
    trials = collect_trials(
        driver,
        lambda: CudaEventClock(torch),
        batch_size=1,
        context_len=64,
        decode_len=DECODE_LEN,
        n_total=N_TOTAL,
    )
    assert len(trials) == N_TOTAL
    for trial in trials:
        assert trial.ttft_ms > 0.0
        assert len(trial.itl_ms) == DECODE_LEN - 1
        assert all(span > 0.0 for span in trial.itl_ms)
