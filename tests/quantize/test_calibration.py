"""The calibration-set builder, on a tiny in-memory dataset with a fake tokenizer."""

from __future__ import annotations

from typing import Any

import datasets
import pytest

from frontier.eval.prompts import ANSWER_TRIGGER, INSTRUCTION
from frontier.quantize.calibration import build_calibration_dataset

N_ROWS = 12
NUM_SAMPLES = 5
MAX_SEQ_LENGTH = 8


class _FakeTokenizer:
    """Records every text it renders and truncates its char-index ids to ``max_length``."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(
        self,
        text: str,
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        assert padding is False
        assert add_special_tokens is False
        self.seen.append(text)
        ids = list(range(len(text)))
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _corpus() -> datasets.Dataset:
    return datasets.Dataset.from_dict(
        {
            "question": [f"question number {i} spelled out at length" for i in range(N_ROWS)],
            "choices": [["alpha", "beta", "gamma", "delta"] for _ in range(N_ROWS)],
            "answer": [i % 4 for i in range(N_ROWS)],
            "subject": ["elementary_mathematics"] * N_ROWS,
        }
    )


def _loader(corpus: datasets.Dataset) -> Any:
    def load(hf_path: str, hf_config: str | None, split: str) -> datasets.Dataset:
        assert (hf_path, hf_config, split) == ("cais/mmlu", "all", "auxiliary_train")
        return corpus

    return load


def _build(tokenizer: _FakeTokenizer, corpus: datasets.Dataset, *, seed: int) -> Any:
    return build_calibration_dataset(
        "in_domain",
        tokenizer,
        num_samples=NUM_SAMPLES,
        max_seq_length=MAX_SEQ_LENGTH,
        seed=seed,
        loader=_loader(corpus),
    )


def test_selects_num_samples_and_drops_text_columns() -> None:
    tokenizer = _FakeTokenizer()
    built = _build(tokenizer, _corpus(), seed=0)
    assert len(built) == NUM_SAMPLES
    assert built.column_names == ["input_ids"]


def test_renders_mcq_prompt_through_build_prompt() -> None:
    tokenizer = _FakeTokenizer()
    _build(tokenizer, _corpus(), seed=0)
    assert len(tokenizer.seen) == NUM_SAMPLES
    for text in tokenizer.seen:
        assert INSTRUCTION in text
        assert "A. alpha" in text
        assert text.endswith(ANSWER_TRIGGER)


def test_truncates_input_ids_to_max_seq_length() -> None:
    tokenizer = _FakeTokenizer()
    built = _build(tokenizer, _corpus(), seed=0)
    assert all(len(ids) <= MAX_SEQ_LENGTH for ids in built["input_ids"])
    assert any(len(ids) == MAX_SEQ_LENGTH for ids in built["input_ids"])


def test_shuffle_is_seed_deterministic() -> None:
    first = _build(_FakeTokenizer(), _corpus(), seed=7)
    again = _build(_FakeTokenizer(), _corpus(), seed=7)
    assert first["input_ids"] == again["input_ids"]


def test_unwired_corpus_raises() -> None:
    with pytest.raises(ValueError, match="is not wired"):
        build_calibration_dataset(
            "ood",
            _FakeTokenizer(),
            num_samples=1,
            max_seq_length=8,
            seed=0,
            loader=_loader(_corpus()),
        )
