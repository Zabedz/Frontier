"""The full config -> score -> metrics -> assemble -> store chain, offline.

A synthetic provider and in-memory records drive the runner with no model and no
network, proving the wiring the live smoke test exercises with a real model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from frontier.eval.records import LETTERS, EvalRecord
from frontier.io.store import ResultStore, read_rows
from frontier.pipeline.runner import run
from frontier.schema import EvalSpec, VariantConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
FP16 = CONFIG_ROOT / "variants" / "fp16.yaml"
GOLD_MARK = "<<GOLD>>"
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
_LETTER_INDEX = {letter: i for i, letter in enumerate(LETTERS)}


class _SyntheticProvider:
    """Favours the gold option, which each record marks in its option text.

    Carries ``backend_version`` so ``_build_backend`` reads a real value off the
    provider, exactly as the HF backend supplies it.
    """

    backend_version = "synthetic-1.0"

    def candidate_token_ids(self, letters: Sequence[str]) -> IntArray:
        return np.arange(len(letters), dtype=np.intp)

    def next_token_logits(self, prompts: Sequence[str]) -> FloatArray:
        return np.stack([self._logits(prompt) for prompt in prompts])

    def _logits(self, prompt: str) -> FloatArray:
        row = np.full(len(LETTERS), -10.0, dtype=np.float64)
        position = 0
        for line in prompt.splitlines():
            if line[1:2] == "." and line[:1] in _LETTER_INDEX:
                row[position] = 2.0 if GOLD_MARK in line else 0.0
                position += 1
        return row


def _records(n: int) -> list[EvalRecord]:
    records = []
    for i in range(n):
        gold = i % 4
        options = tuple(f"choice {j} {GOLD_MARK}" if j == gold else f"choice {j}" for j in range(4))
        records.append(
            EvalRecord(
                qid=f"q{i}", question="Q?", options=options, gold=gold, subject="sub", split="test"
            )
        )
    return records


def test_synthetic_runner_emits_one_valid_row(tmp_path: Path) -> None:
    def factory(variant: VariantConfig, device: str) -> _SyntheticProvider:  # noqa: ARG001
        return _SyntheticProvider()

    def loader(spec: EvalSpec, *, seed: int) -> list[EvalRecord]:  # noqa: ARG001
        return _records(16)

    rows = run(
        FP16,
        mode="smoke",
        config_root=CONFIG_ROOT,
        results_root=tmp_path,
        provider_factory=factory,
        slice_loader=loader,
        timestamp="2026-07-13T00:00:00+00:00",
        git_sha="deadbeef",
    )

    assert len(rows) == 1
    row = rows[0]

    assert row.provenance.git_sha == "deadbeef"
    assert row.provenance.timestamp == "2026-07-13T00:00:00+00:00"
    assert row.provenance.model_id == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert row.provenance.config_hash
    assert row.provenance.hardware_id.startswith("cpu:")

    assert row.backend.inference_backend == "hf"
    assert row.backend.backend_version == "synthetic-1.0"
    assert row.backend.track == "A"

    assert 0.0 <= row.quality.accuracy <= 1.0
    assert np.isfinite(row.quality.ece_equal_width)
    assert 0.0 <= row.quality.ece_equal_width <= 1.0
    assert math.isnan(row.quality.perplexity)

    assert row.robustness is not None
    assert row.latency == []
    assert row.memory == []
    assert math.isnan(row.tok_s_per_gb)

    stored = read_rows(ResultStore(tmp_path))
    assert len(stored) == 1
    assert stored[0].backend.backend_version == "synthetic-1.0"
    assert stored[0].provenance.git_sha == "deadbeef"
    assert math.isnan(stored[0].quality.perplexity)
