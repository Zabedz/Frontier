"""The append-only result store: a durable jsonl log plus a parquet rebuilt from it.

pyarrow rewrites a parquet footer on close and cannot append in place, so each write
appends to the jsonl and rebuilds the parquet atomically (tmp file, then ``Path.replace``).
The store is tens of rows, so the rebuild is cheap. The ``.parquet.tmp`` suffix is
git-ignored; renaming it means editing ``.gitignore`` too.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from frontier.io.serialize import (
    RESULT_COLUMNS,
    RESULT_SCHEMA,
    flatten_record,
    from_record,
    to_record,
    unflatten_record,
)
from frontier.schema import ResultRow

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class ResultStore:
    """Paths for the append-only store rooted at ``root`` (default ``results/``)."""

    root: Path = Path("results")

    @property
    def jsonl_path(self) -> Path:
        return self.root / "results.jsonl"

    @property
    def parquet_path(self) -> Path:
        return self.root / "results.parquet"

    @property
    def parquet_tmp_path(self) -> Path:
        return self.root / "results.parquet.tmp"


def append_row(row: ResultRow, store: ResultStore) -> None:
    """Append one row: jsonl first (durable), then rebuild the parquet atomically."""
    store.root.mkdir(parents=True, exist_ok=True)
    with store.jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_record(row)) + "\n")
    table = _build_table([flatten_record(record) for record in _read_jsonl_records(store)])
    pq.write_table(table, store.parquet_tmp_path)
    store.parquet_tmp_path.replace(store.parquet_path)


def read_rows(store: ResultStore) -> list[ResultRow]:
    """Read the parquet and reconstruct every ``ResultRow`` in insertion order.

    Goes through pyarrow ``to_pylist``, which keeps a null distinct from ``NaN`` so an
    absent ``robustness`` comes back as ``None``.
    """
    table = pq.read_table(store.parquet_path)
    return [from_record(unflatten_record(flat)) for flat in table.to_pylist()]


def read_jsonl_rows(store: ResultStore) -> list[ResultRow]:
    """Read the jsonl mirror and reconstruct every ``ResultRow``."""
    return [from_record(record) for record in _read_jsonl_records(store)]


def read_frame(store: ResultStore) -> pd.DataFrame:
    """The flat parquet as a pandas ``DataFrame``, for analysis and plotting."""
    frame: pd.DataFrame = pq.read_table(store.parquet_path).to_pandas()
    return frame


def _build_table(flat_rows: list[Mapping[str, Any]]) -> pa.Table:
    columns = {name: [row[name] for row in flat_rows] for name in RESULT_COLUMNS}
    return pa.Table.from_pydict(columns, schema=RESULT_SCHEMA)


def _read_jsonl_records(store: ResultStore) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with store.jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records
