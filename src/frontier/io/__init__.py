"""The append-only result store (parquet plus a jsonl mirror), its per-item prediction
sidecars, and provenance stamping. Analysis and plotting read results from here.
"""

from __future__ import annotations

from frontier.io.predictions import (
    PREDICTIONS_SUBDIR,
    PredictionRows,
    predictions_key,
    predictions_path,
    read_predictions,
    read_predictions_path,
    write_predictions_rows,
)
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
    "PREDICTIONS_SUBDIR",
    "RESULT_COLUMNS",
    "RESULT_SCHEMA",
    "HardwareInfo",
    "PredictionRows",
    "ResultStore",
    "append_row",
    "flatten_record",
    "from_record",
    "hardware_info",
    "now_utc_iso",
    "predictions_key",
    "predictions_path",
    "read_frame",
    "read_git_sha",
    "read_jsonl_rows",
    "read_predictions",
    "read_predictions_path",
    "read_rows",
    "stamp_provenance",
    "to_record",
    "unflatten_record",
    "write_predictions_rows",
]
