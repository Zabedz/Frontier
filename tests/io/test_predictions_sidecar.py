"""Sidecar round-trip, the write-time guards, and reading a sidecar without distributions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from frontier.io.predictions import (
    OptionProbs,
    PredictionRows,
    pad_option_probs,
    predictions_path,
    read_predictions_path,
    write_predictions_rows,
)

KEY = "mmlu__abc__seed0"


def _rows(option_counts: tuple[int, ...] = (4, 4, 3), *, seed: int = 0) -> PredictionRows:
    rng = np.random.default_rng(seed)
    draws = [rng.dirichlet(np.ones(count)) for count in option_counts]
    options = pad_option_probs([np.asarray(draw, dtype=np.float64) for draw in draws])
    predicted = np.asarray([int(draw.argmax()) for draw in draws], dtype=np.intp)
    confidence = np.asarray([float(draw.max()) for draw in draws], dtype=np.float64)
    gold = np.asarray([0, 1, 2], dtype=np.intp)
    return PredictionRows(
        confidence=confidence,
        correct=(predicted == gold),
        gold=gold,
        predicted=predicted,
        options=options,
    )


def test_round_trip_preserves_the_distributions_and_their_option_counts(tmp_path: Path) -> None:
    rows = _rows()
    assert rows.options is not None
    path = write_predictions_rows(rows, root=tmp_path, key=KEY)
    back = read_predictions_path(path)
    assert back.options is not None
    assert np.array_equal(back.options.n_options, rows.options.n_options)
    assert np.allclose(back.options.probs, rows.options.probs, atol=0.0)
    assert np.allclose(back.confidence, rows.confidence, atol=0.0)
    assert np.array_equal(back.predicted, rows.predicted)


def test_the_stored_file_holds_no_padding(tmp_path: Path) -> None:
    """Each item is stored at its true option count, so no 0.0 cell reaches a later log."""
    path = write_predictions_rows(_rows((4, 4, 3)), root=tmp_path, key=KEY)
    stored = pq.read_table(path).column("probs").to_pylist()
    assert [len(item) for item in stored] == [4, 4, 3]


def test_a_sidecar_without_distributions_reads_back_as_none(tmp_path: Path) -> None:
    """The six banked sidecars predate the column and must still load."""
    legacy = pa.table(
        {
            "confidence": pa.array([0.9, 0.7], pa.float64()),
            "correct": pa.array([True, False], pa.bool_()),
            "gold": pa.array([0, 1], pa.int64()),
            "predicted": pa.array([0, 2], pa.int64()),
        }
    )
    path = predictions_path(tmp_path, KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(legacy, path)
    back = read_predictions_path(path)
    assert back.options is None
    assert back.confidence.shape == (2,)


def test_writing_without_distributions_round_trips_as_none(tmp_path: Path) -> None:
    rows = replace(_rows(), options=None)
    back = read_predictions_path(write_predictions_rows(rows, root=tmp_path, key=KEY))
    assert back.options is None
    assert np.allclose(back.confidence, rows.confidence, atol=0.0)


def test_a_distribution_disagreeing_with_predicted_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    broken = replace(rows, predicted=np.asarray([1, 0, 0], dtype=np.intp))
    with pytest.raises(ValueError, match="peaks at option"):
        write_predictions_rows(broken, root=tmp_path, key=KEY)


def test_a_distribution_disagreeing_with_confidence_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    broken = replace(rows, confidence=rows.confidence * 0.5)
    with pytest.raises(ValueError, match="peaks at"):
        write_predictions_rows(broken, root=tmp_path, key=KEY)


def test_a_distribution_off_the_simplex_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    assert rows.options is not None
    probs = rows.options.probs.copy()
    probs[1] *= 0.5
    broken = replace(rows, options=OptionProbs(probs=probs, n_options=rows.options.n_options))
    with pytest.raises(ValueError, match="sums to"):
        write_predictions_rows(broken, root=tmp_path, key=KEY)


@pytest.mark.parametrize("count", [0, 5])
def test_an_out_of_range_option_count_is_rejected(tmp_path: Path, count: int) -> None:
    rows = _rows()
    assert rows.options is not None
    counts = rows.options.n_options.copy()
    counts[0] = count
    broken = replace(rows, options=OptionProbs(probs=rows.options.probs, n_options=counts))
    with pytest.raises(ValueError, match="outside"):
        write_predictions_rows(broken, root=tmp_path, key=KEY)


def test_a_distribution_of_the_wrong_length_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    assert rows.options is not None
    broken = replace(
        rows,
        options=OptionProbs(probs=rows.options.probs[:2], n_options=rows.options.n_options[:2]),
    )
    with pytest.raises(ValueError, match="disagree in length"):
        write_predictions_rows(broken, root=tmp_path, key=KEY)


def test_a_non_finite_distribution_is_rejected(tmp_path: Path) -> None:
    """NaN fails every comparison, and an all-NaN row's argmax agrees with a predicted
    taken from that same row, so the finite check has to come first.

    Reachable: the vLLM provider seeds rows with -inf and overwrites only the letters the
    backend returned, so a missing letter softmaxes to NaN.
    """
    rows = _rows()
    assert rows.options is not None
    probs = rows.options.probs.copy()
    probs[0, :] = np.nan
    confidence = rows.confidence.copy()
    confidence[0] = np.nan
    broken = replace(
        rows,
        confidence=confidence,
        options=OptionProbs(probs=probs, n_options=rows.options.n_options),
    )
    with pytest.raises(ValueError, match="not finite"):
        write_predictions_rows(broken, root=tmp_path, key=KEY)


def test_a_partly_filled_distribution_column_is_rejected(tmp_path: Path) -> None:
    """The writer is all-or-nothing, so a mixed column means the file is corrupt."""
    mixed = pa.table(
        {
            "confidence": pa.array([0.9, 0.7], pa.float64()),
            "correct": pa.array([True, False], pa.bool_()),
            "gold": pa.array([0, 1], pa.int64()),
            "predicted": pa.array([0, 1], pa.int64()),
            "probs": pa.array([[0.9, 0.1], None], pa.list_(pa.float64())),
        }
    )
    path = predictions_path(tmp_path, KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(mixed, path)
    with pytest.raises(ValueError, match="corrupt"):
        read_predictions_path(path)
