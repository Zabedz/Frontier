"""Predictions sidecar: round-trip, key derivation, and the length guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frontier.io.predictions import (
    PREDICTIONS_SUBDIR,
    PredictionRows,
    predictions_key,
    predictions_path,
    read_predictions,
    write_predictions_rows,
)


def _rows(n: int) -> PredictionRows:
    rng = np.random.default_rng(0)
    confidence = rng.uniform(0.25, 1.0, size=n)
    correct = rng.uniform(0.0, 1.0, size=n) < confidence
    gold = rng.integers(0, 4, size=n).astype(np.intp)
    predicted = np.where(correct, gold, (gold + 1) % 4).astype(np.intp)
    return PredictionRows(confidence, correct.astype(np.bool_), gold, predicted, options=None)


def test_round_trip_preserves_values_and_dtypes(tmp_path: Path) -> None:
    original = _rows(37)
    key = predictions_key("abc123", 0, "mmlu")
    written = write_predictions_rows(original, root=tmp_path, key=key)
    assert written == predictions_path(tmp_path, key)

    loaded = read_predictions(tmp_path, key)
    assert np.array_equal(loaded.confidence, original.confidence)
    assert np.array_equal(loaded.correct, original.correct)
    assert np.array_equal(loaded.gold, original.gold)
    assert np.array_equal(loaded.predicted, original.predicted)
    assert loaded.confidence.dtype == np.float64
    assert loaded.correct.dtype == np.bool_
    assert loaded.gold.dtype == np.intp
    assert loaded.predicted.dtype == np.intp


def test_key_is_deterministic_and_path_lands_under_the_subdir(tmp_path: Path) -> None:
    key = predictions_key("hash0", 2, "arc_challenge")
    assert key == predictions_key("hash0", 2, "arc_challenge")
    assert key == "arc_challenge__hash0__seed2"

    path = predictions_path(tmp_path, key)
    assert path.parent.name == PREDICTIONS_SUBDIR
    assert path.name == "arc_challenge__hash0__seed2.parquet"


def test_length_mismatch_raises_naming_the_lengths(tmp_path: Path) -> None:
    bad = PredictionRows(
        confidence=np.array([0.9, 0.8], dtype=np.float64),
        correct=np.array([True], dtype=np.bool_),
        gold=np.array([0, 1], dtype=np.intp),
        predicted=np.array([0, 1], dtype=np.intp),
        options=None,
    )
    with pytest.raises(ValueError, match="length"):
        write_predictions_rows(bad, root=tmp_path, key="k")
