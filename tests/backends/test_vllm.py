"""The vLLM extraction, driven by a fake engine and tokenizer on CPU.

Proves the correctness backbone: the provider reads only the resolved letter ids out of
the full-vocab logprob dict, the candidate softmax equals softmax of the injected
candidate logprobs, and that softmax is invariant to a shared additive constant added to
every returned logprob (which is why full-vocab and subset-masked logprobs are
equivalent). It also pins the SamplingParams the provider passes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from frontier.backends.vllm import VllmLogitProvider
from frontier.eval.prompts import build_prompt
from frontier.eval.provider import letter_probs, softmax

VOCAB = 200
LETTER_IDS = {" A": 10, " B": 11, " C": 12, " D": 13}
# Two non-letter ids that are present in the returned dict at a competitive rank; the
# provider must ignore them, because only the letters carry the answer signal.
DISTRACTOR_IDS = {5: -8.0, 99: -0.2}
CANDIDATE_LOGPROBS = {10: -1.0, 11: -2.0, 12: -0.5, 13: -3.0}


class _FakeSamplingParams:
    def __init__(self, *, temperature: float, max_tokens: int, logprobs: int, seed: int) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        self.seed = seed


class _FakeLogprob:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


class _FakeCompletion:
    def __init__(self, logprobs: dict[int, _FakeLogprob]) -> None:
        self.logprobs = [logprobs]


class _FakeRequestOutput:
    def __init__(self, logprobs: dict[int, _FakeLogprob]) -> None:
        self.outputs = [_FakeCompletion(logprobs)]


class _FakeEngine:
    def __init__(self, per_prompt: Sequence[dict[int, float]]) -> None:
        self._per_prompt = per_prompt
        self.captured_sampling_params: Any = None
        self.captured_prompts: Any = None

    def generate(self, prompts: Any, sampling_params: Any) -> list[_FakeRequestOutput]:
        self.captured_prompts = prompts
        self.captured_sampling_params = sampling_params
        return [
            _FakeRequestOutput({tid: _FakeLogprob(lp) for tid, lp in row.items()})
            for row in self._per_prompt
        ]


class _FakeTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,  # noqa: ARG002
        add_generation_prompt: bool,  # noqa: ARG002
    ) -> str:
        return f"<|im_start|>user\n{conversation[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        if text in LETTER_IDS:
            return [LETTER_IDS[text]]
        return [len(text)]

    def __len__(self) -> int:
        return VOCAB


def _provider(per_prompt: Sequence[dict[int, float]]) -> tuple[VllmLogitProvider, _FakeEngine]:
    engine = _FakeEngine(per_prompt)
    provider = VllmLogitProvider(
        model="checkpoint-dir",
        tokenizer_id="Qwen/Qwen2.5-3B-Instruct",
        device="cuda",
        weight_dtype="int4",
        max_letters=4,
        engine=engine,
        tokenizer=_FakeTokenizer(),
        sampling_params_cls=_FakeSamplingParams,
    )
    return provider, engine


def _row_dict() -> dict[int, float]:
    return {**CANDIDATE_LOGPROBS, **DISTRACTOR_IDS}


def test_reads_only_letter_ids_into_a_neg_inf_row() -> None:
    provider, _ = _provider([_row_dict()])
    row = provider.next_token_logits([build_prompt("Q?", ("a", "b", "c", "d"))])
    assert row.shape == (1, VOCAB)
    for token_id, logprob in CANDIDATE_LOGPROBS.items():
        assert row[0, token_id] == logprob
    for token_id in DISTRACTOR_IDS:
        assert np.isneginf(row[0, token_id])
    assert np.isneginf(row[0, 0])


def test_candidate_softmax_equals_softmax_of_injected_logprobs() -> None:
    provider, _ = _provider([_row_dict()])
    row = provider.next_token_logits([build_prompt("Q?", ("a", "b", "c", "d"))])
    candidate_ids = provider.candidate_token_ids("ABCD")
    assert list(candidate_ids) == [10, 11, 12, 13]
    probs = letter_probs(row[0], candidate_ids)
    expected = softmax(np.array([-1.0, -2.0, -0.5, -3.0]))
    assert np.allclose(probs, expected)


def test_confidence_is_invariant_to_a_shared_additive_constant() -> None:
    shift = 5.0
    base_provider, _ = _provider([_row_dict()])
    base_row = base_provider.next_token_logits([build_prompt("Q?", ("a", "b", "c", "d"))])
    base = letter_probs(base_row[0], base_provider.candidate_token_ids("ABCD"))

    shifted = {tid: lp + shift for tid, lp in _row_dict().items()}
    shift_provider, _ = _provider([shifted])
    shift_row = shift_provider.next_token_logits([build_prompt("Q?", ("a", "b", "c", "d"))])
    shifted_probs = letter_probs(shift_row[0], shift_provider.candidate_token_ids("ABCD"))

    assert np.allclose(base, shifted_probs)


def test_sampling_params_are_temperature_one_full_vocab_single_token() -> None:
    provider, engine = _provider([_row_dict()])
    provider.next_token_logits([build_prompt("Q?", ("a", "b", "c", "d"))])
    params = engine.captured_sampling_params
    assert params.temperature == 1.0
    assert params.logprobs == -1
    assert params.max_tokens == 1
    assert getattr(params, "allowed_token_ids", None) is None


def test_generate_is_fed_token_id_prompts() -> None:
    provider, engine = _provider([_row_dict()])
    provider.next_token_logits([build_prompt("Q?", ("a", "b", "c", "d"))])
    assert "prompt_token_ids" in engine.captured_prompts[0]


def test_model_is_public_for_the_fp16_gate() -> None:
    provider, _ = _provider([_row_dict()])
    assert provider.model == "checkpoint-dir"


def test_more_letters_than_max_letters_raises() -> None:
    provider, _ = _provider([_row_dict()])
    with pytest.raises(ValueError, match="raise max_letters"):
        provider.candidate_token_ids("ABCDE")
