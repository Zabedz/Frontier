"""The llama.cpp extraction, driven by a fake Llama and tokenizer on CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frontier.backends.llama_cpp import LlamaCppLogitProvider
from frontier.eval.prompts import build_prompt

VOCAB = 64
N_CTX = 16
PROMPT_LEN = 7


class _FakeLlama:
    """A llama.cpp stand-in: table-driven tokenisation and a filled scores buffer."""

    def __init__(
        self, letter_table: dict[bytes, list[int]], *, prompt_len: int = PROMPT_LEN
    ) -> None:
        self._letter_table = letter_table
        self._prompt_len = prompt_len
        self.reset_calls = 0
        self._n_tokens = 0
        self.scores = np.arange(N_CTX * VOCAB, dtype=np.float32).reshape(N_CTX, VOCAB)

    def tokenize(self, text: bytes, add_bos: bool, special: bool) -> list[int]:
        assert add_bos is False
        if special:
            return list(range(self._prompt_len))
        return list(self._letter_table[text])

    def reset(self) -> None:
        self.reset_calls += 1

    def eval(self, tokens: list[int]) -> None:
        self._n_tokens = len(tokens)

    @property
    def n_tokens(self) -> int:
        return self._n_tokens


class _ChatTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,  # noqa: ARG002
        add_generation_prompt: bool,  # noqa: ARG002
    ) -> str:
        return f"<|im_start|>user\n{conversation[0]['content']}<|im_end|>\n<|im_start|>assistant\n"


def _provider(llama: _FakeLlama) -> LlamaCppLogitProvider:
    return LlamaCppLogitProvider(
        gguf_path=Path("model.gguf"),
        tokenizer_id="Qwen/Qwen2.5-3B-Instruct",
        device="cuda",
        n_gpu_layers=99,
        weight_dtype="q4_k_m",
        llama=llama,
        tokenizer=_ChatTokenizer(),
    )


def _spaced_table() -> dict[bytes, list[int]]:
    return {b" A": [30], b" B": [31], b" C": [32], b" D": [33]}


def test_next_token_logits_reads_the_last_position() -> None:
    llama = _FakeLlama(_spaced_table())
    provider = _provider(llama)
    row = provider.next_token_logits([build_prompt("Q?", ("a", "b", "c", "d"))])
    assert row.shape == (1, VOCAB)
    assert np.array_equal(row[0], llama.scores[PROMPT_LEN - 1])


def test_reset_is_called_once_per_prompt() -> None:
    llama = _FakeLlama(_spaced_table())
    provider = _provider(llama)
    prompts = [build_prompt("Q1?", ("a", "b")), build_prompt("Q2?", ("a", "b"))]
    provider.next_token_logits(prompts)
    assert llama.reset_calls == len(prompts)


def test_candidate_ids_resolve_against_the_llama_vocab() -> None:
    provider = _provider(_FakeLlama(_spaced_table()))
    ids = provider.candidate_token_ids("ABCD")
    assert list(ids) == [30, 31, 32, 33]
    assert ids.dtype == np.intp


def test_candidate_resolution_falls_back_to_bare_letter() -> None:
    provider = _provider(_FakeLlama({b" A": [1, 2], b"A": [40]}))
    assert list(provider.candidate_token_ids("A")) == [40]


def test_candidate_resolution_raises_when_both_split() -> None:
    provider = _provider(_FakeLlama({b" A": [1, 2], b"A": [3, 4]}))
    with pytest.raises(ValueError, match="not a single llama token"):
        provider.candidate_token_ids("A")
