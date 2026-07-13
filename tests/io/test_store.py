"""Append-only store: parquet + jsonl reconstruct the rows, atomically."""

from __future__ import annotations

import math
from pathlib import Path

from rows import nan_row, sample_row

from frontier.io.serialize import RESULT_COLUMNS
from frontier.io.store import ResultStore, append_row, read_frame, read_jsonl_rows, read_rows


def test_append_and_read_both_mirrors(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    rows = [sample_row(seed=0), sample_row(seed=1, with_robustness=False)]
    for row in rows:
        append_row(row, store)

    assert read_rows(store) == rows
    assert read_jsonl_rows(store) == rows
    assert read_rows(store)[1].robustness is None

    assert store.jsonl_path.read_text().count("\n") == len(rows)
    assert store.parquet_path.exists()
    assert not list(tmp_path.glob("*.parquet.tmp"))


def test_append_is_additive(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    seeds = [0, 1, 2]
    for seed in seeds:
        append_row(sample_row(seed=seed), store)
    assert len(read_rows(store)) == len(seeds)
    assert [row.provenance.seed for row in read_rows(store)] == seeds


def test_read_frame_is_flat_with_all_columns(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    rows = [sample_row(seed=0), sample_row(seed=1)]
    for row in rows:
        append_row(row, store)
    frame = read_frame(store)
    assert frame.shape[0] == len(rows)
    assert list(frame.columns) == list(RESULT_COLUMNS)


def test_nan_row_survives_the_store(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    append_row(nan_row(), store)
    restored = read_rows(store)[0]
    assert math.isnan(restored.quality.perplexity)
    assert math.isnan(restored.tok_s_per_gb)


def test_absent_robustness_reconstructs_as_none_from_parquet(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    append_row(sample_row(with_robustness=False), store)
    assert read_rows(store)[0].robustness is None
