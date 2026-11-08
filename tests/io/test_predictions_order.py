"""The per-order distribution column in the predictions sidecar."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from frontier.analysis.holdout import take
from frontier.io.predictions import (
    ORDER_PROBS_COLUMN,
    CorruptSidecarError,
    OrderTensor,
    PredictionRows,
    pad_option_probs,
    read_predictions_path,
    write_predictions_rows,
)

KEY = "mmlu__abc__seed0"


def _pool(block: OrderTensor) -> OrderTensor:
    logits = np.log(block).mean(axis=0)
    pooled: OrderTensor = np.exp(logits - logits.max())
    pooled /= pooled.sum()
    return pooled


def _rows(option_counts: tuple[int, ...] = (4, 4, 3), *, seed: int = 0) -> PredictionRows:
    rng = np.random.default_rng(seed)
    width = max(option_counts)
    tensor = np.zeros((len(option_counts), width, width), dtype=np.float64)
    pooled = []
    for index, count in enumerate(option_counts):
        block = rng.dirichlet(np.ones(count), size=count)
        tensor[index, :count, :count] = block
        pooled.append(_pool(block))
    options = pad_option_probs(pooled)
    predicted = np.asarray([int(row.argmax()) for row in pooled], dtype=np.intp)
    gold = np.zeros(len(option_counts), dtype=np.intp)
    return PredictionRows(
        confidence=np.asarray([float(row.max()) for row in pooled], dtype=np.float64),
        correct=predicted == gold,
        gold=gold,
        predicted=predicted,
        options=options,
        qid=np.asarray([f"q{i}" for i in range(len(option_counts))], dtype=np.str_),
        order_probs=tensor,
    )


def test_round_trip_keeps_each_order_at_its_true_size(tmp_path: Path) -> None:
    rows = _rows()
    assert rows.order_probs is not None
    path = write_predictions_rows(rows, root=tmp_path, key=KEY)
    stored = pq.read_table(path).column(ORDER_PROBS_COLUMN).to_pylist()
    assert [len(item) for item in stored] == [16, 16, 9]
    back = read_predictions_path(path)
    assert back.order_probs is not None
    assert np.array_equal(back.order_probs, rows.order_probs)


def test_a_sidecar_without_the_column_reads_back_as_none(tmp_path: Path) -> None:
    rows = replace(_rows(), order_probs=None)
    path = write_predictions_rows(rows, root=tmp_path, key=KEY)
    legacy = pq.read_table(path).drop_columns([ORDER_PROBS_COLUMN])
    pq.write_table(legacy, path)
    assert read_predictions_path(path).order_probs is None


def test_orders_that_do_not_pool_to_the_stored_distribution_are_rejected(
    tmp_path: Path,
) -> None:
    rows = _rows()
    assert rows.order_probs is not None
    swapped = rows.order_probs.copy()
    swapped[0] = swapped[0, :, ::-1]
    with pytest.raises(ValueError, match="pool to"):
        write_predictions_rows(replace(rows, order_probs=swapped), root=tmp_path, key=KEY)


def test_an_order_off_the_simplex_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    assert rows.order_probs is not None
    scaled = rows.order_probs.copy()
    scaled[1, 2] *= 0.5
    with pytest.raises(ValueError, match="sum to"):
        write_predictions_rows(replace(rows, order_probs=scaled), root=tmp_path, key=KEY)


def test_orders_without_the_pooled_distribution_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pooled distribution beside them"):
        write_predictions_rows(replace(_rows(), options=None), root=tmp_path, key=KEY)


def test_take_carries_the_orders_along() -> None:
    rows = _rows()
    assert rows.order_probs is not None
    picked = take(rows, np.asarray([2, 0], dtype=np.intp))
    assert picked.order_probs is not None
    assert np.array_equal(picked.order_probs, rows.order_probs[[2, 0]])


def test_a_stored_block_at_the_wrong_size_is_corrupt(tmp_path: Path) -> None:
    rows = _rows((4, 4, 3))
    path = write_predictions_rows(rows, root=tmp_path, key=KEY)
    table = pq.read_table(path)
    stored = table.column(ORDER_PROBS_COLUMN).to_pylist()
    stored[0] = stored[0][:9]
    index = table.column_names.index(ORDER_PROBS_COLUMN)
    column = pa.array(stored, pa.list_(pa.float64()))
    pq.write_table(table.set_column(index, ORDER_PROBS_COLUMN, column), path)
    with pytest.raises(CorruptSidecarError, match="beside 4 options"):
        read_predictions_path(path)
