"""The `frontier recalibrate` command over a seeded store."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from frontier.io.predictions import (
    PredictionRows,
    ProbMatrix,
    pad_option_probs,
    predictions_key,
    write_predictions_rows,
)
from frontier.io.store import ResultStore, append_row
from frontier.pipeline.cli import app

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))

from rows import sample_row

N_ITEMS = 2000
USAGE_ERROR = 2  # click's exit code for a bad parameter


def _seeded_store(root: Path) -> None:
    rng = np.random.default_rng(3)
    logits = rng.normal(0.0, 2.0, size=(N_ITEMS, 4))

    def softmax(scores: ProbMatrix) -> ProbMatrix:
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        normalised: ProbMatrix = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        return normalised

    gold = np.asarray([rng.choice(4, p=row) for row in softmax(logits)], dtype=np.intp)
    store = ResultStore(root)
    probs = softmax(logits / 0.45)
    predicted = probs.argmax(axis=1).astype(np.intp)
    base = sample_row()
    append_row(
        replace(
            base,
            variant_name="int4-nf4",
            provenance=replace(base.provenance, config_hash="a" * 64),
        ),
        store,
    )
    write_predictions_rows(
        PredictionRows(
            confidence=probs.max(axis=1),
            correct=(predicted == gold),
            gold=gold,
            predicted=predicted,
            options=pad_option_probs(list(probs)),
            qid=np.asarray([f"q{index}" for index in range(N_ITEMS)], dtype=np.str_),
        ),
        root=root,
        key=predictions_key("a" * 64, 0, "mmlu"),
    )


def test_recalibrate_writes_the_table_and_leaves_no_staging_file(tmp_path: Path) -> None:
    _seeded_store(tmp_path)
    result = CliRunner().invoke(app, ["recalibrate", "--results", str(tmp_path)])
    assert result.exit_code == 0, result.output
    destination = tmp_path / "recalibration.parquet"
    assert destination.exists()
    assert not destination.with_suffix(".parquet.tmp").exists()
    frame = pd.read_parquet(destination)
    assert list(frame["variant"]) == ["int4-nf4"]
    assert frame["ece_after"].iloc[0] < frame["ece_before"].iloc[0]


def test_recalibrate_writes_nothing_when_no_variant_carries_distributions(
    tmp_path: Path,
) -> None:
    store = ResultStore(tmp_path)
    append_row(sample_row(), store)
    result = CliRunner().invoke(app, ["recalibrate", "--results", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "recalibration.parquet").exists()


def test_a_reference_map_can_be_pointed_at(tmp_path: Path) -> None:
    _seeded_store(tmp_path)
    references = tmp_path / "references.yaml"
    references.write_text("references:\n  hf: fp16\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["recalibrate", "--results", str(tmp_path), "--references", str(references)],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "recalibration.parquet").exists()


def test_a_missing_reference_map_is_a_usage_error_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default path is repo-relative, so the command has to say so from elsewhere."""
    _seeded_store(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["recalibrate", "--results", str(tmp_path)])
    assert result.exit_code == USAGE_ERROR, result.output
    assert "--references" in result.output
    assert not isinstance(result.exception, FileNotFoundError)
