"""Live llama.cpp checks: a real GGUF gives finite logits and single-token answer letters."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("llama_cpp")
pytest.importorskip("transformers")

from frontier.backends.llama_cpp import LlamaCppLogitProvider
from frontier.eval.prompts import build_prompt

QWEN = "Qwen/Qwen2.5-3B-Instruct"
N_LETTERS = 4

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("FRONTIER_LIVE_MODELS") or not os.environ.get("FRONTIER_GGUF_PATH"),
        reason="live GGUF; set FRONTIER_LIVE_MODELS=1 and FRONTIER_GGUF_PATH to run",
    ),
]


@pytest.fixture(scope="module")
def provider() -> LlamaCppLogitProvider:
    return LlamaCppLogitProvider(
        gguf_path=Path(os.environ["FRONTIER_GGUF_PATH"]),
        tokenizer_id=QWEN,
        device="cpu",
        n_gpu_layers=0,
        weight_dtype="q4_k_m",
        n_ctx=2048,
    )


def test_letters_resolve_to_distinct_single_tokens(provider: LlamaCppLogitProvider) -> None:
    ids = provider.candidate_token_ids("ABCD")
    assert ids.shape == (N_LETTERS,)
    assert len(set(ids.tolist())) == N_LETTERS


def test_next_token_logits_are_finite(provider: LlamaCppLogitProvider) -> None:
    prompts = [
        build_prompt("The capital of France is", ("Paris", "Rome", "Berlin", "Madrid")),
        build_prompt("2 plus 2 equals", ("3", "4", "5", "6")),
    ]
    logits = provider.next_token_logits(prompts)
    assert logits.shape[0] == len(prompts)
    candidate_ids = provider.candidate_token_ids("ABCD")
    assert np.all(np.isfinite(logits[:, candidate_ids]))
    assert provider.backend_version != "unknown"
