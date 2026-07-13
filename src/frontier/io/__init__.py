"""The result store: append-only parquet with a jsonl mirror, plus provenance
stamping (git SHA, config hash, model revision, hardware id, driver + CUDA
version). Analysis and plotting read from here; no consumer recomputes from raw
model outputs.
"""

from __future__ import annotations

from frontier.io.provenance import (
    HardwareInfo,
    hardware_info,
    now_utc_iso,
    read_git_sha,
    stamp_provenance,
)
from frontier.io.serialize import (
    RESULT_COLUMNS,
    RESULT_SCHEMA,
    flatten_record,
    from_record,
    to_record,
    unflatten_record,
)
from frontier.io.store import (
    ResultStore,
    append_row,
    read_frame,
    read_jsonl_rows,
    read_rows,
)

__all__ = [
    "RESULT_COLUMNS",
    "RESULT_SCHEMA",
    "HardwareInfo",
    "ResultStore",
    "append_row",
    "flatten_record",
    "from_record",
    "hardware_info",
    "now_utc_iso",
    "read_frame",
    "read_git_sha",
    "read_jsonl_rows",
    "read_rows",
    "stamp_provenance",
    "to_record",
    "unflatten_record",
]
