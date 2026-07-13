"""Live SmolLM2-135M checks on CPU. Excluded from the default run.

Marked ``slow`` and gated on both an installed ``transformers``/``torch`` (the ``hf``
group) and ``FRONTIER_LIVE_MODELS``, so the default offline ``uv run pytest`` skips it
before touching the network. It confirms the real tokenizer resolves the answer
letters via the leading-space path and that a forward pass yields finite logits.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from frontier.backends.hf import HFLogitProvider
from frontier.eval.prompts import build_prompt

SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOL_VOCAB = 49152
N_LETTERS = 4

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("FRONTIER_LIVE_MODELS"),
        reason="live model download; set FRONTIER_LIVE_MODELS=1 to run",
    ),
]


@pytest.fixture(scope="module")
def provider() -> HFLogitProvider:
    return HFLogitProvider(model_id=SMOL, device="cpu", weight_dtype="fp16")


def test_candidate_ids_are_four_distinct_single_tokens(provider: HFLogitProvider) -> None:
    ids = provider.candidate_token_ids("ABCD")
    assert ids.shape == (N_LETTERS,)
    assert len(set(ids.tolist())) == N_LETTERS


def test_next_token_logits_shape_and_finite(provider: HFLogitProvider) -> None:
    prompts = [
        build_prompt("What is the capital of France?", ("Paris", "Rome", "Berlin", "Madrid")),
        build_prompt("2 + 2 = ?", ("3", "4", "5", "6")),
    ]
    logits = provider.next_token_logits(prompts)
    assert logits.shape == (2, SMOL_VOCAB)
    assert np.all(np.isfinite(logits))
    assert provider.backend_version != "unknown"
