"""The model-free parts of the HF backend: wrapping, resolution, device/dtype."""

from __future__ import annotations

import numpy as np
import pytest

from frontier.backends.hf import chat_wrap, resolve_candidates, resolve_device, resolve_dtype
from frontier.eval.prompts import ANSWER_TRIGGER, build_prompt


class _ChatFake:
    """Echoes the chat template so the assistant-prefill wrapping is observable."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,  # noqa: ARG002
        add_generation_prompt: bool,  # noqa: ARG002
    ) -> str:
        content = conversation[0]["content"]
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


class _EncodeFake:
    """A tokenizer backed by an explicit text -> ids table (encode only)."""

    def __init__(self, table: dict[str, list[int]]) -> None:
        self._table = table

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        return list(self._table[text])


def test_chat_wrap_keeps_answer_trigger_as_prefill() -> None:
    prompt = build_prompt("What is 2+2?", ("3", "4", "5", "6"))
    wrapped = chat_wrap(_ChatFake(), prompt)
    assert wrapped.endswith(f"<|im_start|>assistant\n{ANSWER_TRIGGER}")
    assert "What is 2+2?" in wrapped


def test_chat_wrap_rejects_non_trigger_prompt() -> None:
    cot = build_prompt("What is 2+2?", ("3", "4"), cot=True)
    with pytest.raises(ValueError, match="Answer:"):
        chat_wrap(_ChatFake(), cot)


def test_resolve_candidates_uses_leading_space_when_single_token() -> None:
    tokenizer = _EncodeFake({" A": [10], " B": [11], " C": [12], " D": [13]})
    ids = resolve_candidates(tokenizer, "ABCD")
    assert list(ids) == [10, 11, 12, 13]


def test_resolve_candidates_falls_back_to_bare_letter() -> None:
    tokenizer = _EncodeFake(
        {
            " A": [1, 2],
            " B": [3, 4],
            "A": [10],
            "B": [11],
        }
    )
    ids = resolve_candidates(tokenizer, "AB")
    assert list(ids) == [10, 11]


def test_resolve_candidates_propagates_when_both_split() -> None:
    tokenizer = _EncodeFake({" A": [1, 2], "A": [3, 4]})
    with pytest.raises(ValueError, match="single token"):
        resolve_candidates(tokenizer, "A")


def test_resolve_device() -> None:
    assert resolve_device("smoke") == "cpu"
    assert resolve_device("full", cuda_available=False) == "cpu"
    assert resolve_device("full", cuda_available=True) == "cuda"


def test_resolve_dtype() -> None:
    assert resolve_dtype("cpu", "fp16") == "float32"
    assert resolve_dtype("cpu", "bf16") == "float32"
    assert resolve_dtype("cuda", "fp16") == "float16"
    assert resolve_dtype("cuda", "bf16") == "bfloat16"


def test_resolve_dtype_rejects_unknown_cuda_dtype() -> None:
    with pytest.raises(ValueError, match="no torch compute dtype"):
        resolve_dtype("cuda", "int4")


def test_candidate_ids_dtype_is_intp() -> None:
    tokenizer = _EncodeFake({" A": [10], " B": [11]})
    ids = resolve_candidates(tokenizer, "AB")
    assert ids.dtype == np.intp
