"""Flatten a nested ``ResultRow`` to a columnar row and back.

The store keeps two mirrors of every row: a nested jsonl line (``to_record`` /
``from_record``) and a wide parquet row (``flatten_record`` / ``unflatten_record``).
Three field kinds need care:

- Nested-record scalars flatten to dotted columns (``provenance.git_sha``,
  ``quality.accuracy``) with parquet-native types.
- ``quality.ece_bin_sweep`` (a ``dict[int, float]``) and the two variable-length
  lists ``latency`` and ``memory`` serialize to one JSON-string column each, so a
  populated ``latency``/``memory`` round-trips with no schema change once the latency
  rig lands.
- ``robustness`` flattens to three nullable float columns. Absence is a genuine SQL
  null, not ``NaN``, so the reader reconstructs ``None`` exactly when
  ``robustness.permutation_consistency`` is null. Read-back therefore goes through the
  pyarrow ``to_pylist`` layer, which keeps null distinct from ``NaN``, not through a
  pandas dtype.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

import pyarrow as pa

from frontier.schema import (
    Backend,
    Latency,
    MachineState,
    Memory,
    Provenance,
    Quality,
    ResultRow,
    Robustness,
    TaskSpec,
)

_PROVENANCE_FIELDS = tuple(f.name for f in dataclasses.fields(Provenance))
_BACKEND_FIELDS = tuple(f.name for f in dataclasses.fields(Backend))
_TASK_FIELDS = tuple(f.name for f in dataclasses.fields(TaskSpec))
_QUALITY_SCALAR_FIELDS = tuple(
    f.name for f in dataclasses.fields(Quality) if f.name != "ece_bin_sweep"
)
_ROBUSTNESS_FIELDS = tuple(f.name for f in dataclasses.fields(Robustness))

_INT_COLUMNS = frozenset({"provenance.seed", "backend.gpu_offload_layers", "task.num_items"})
_BOOL_COLUMNS = frozenset({"task.cot", "quality.temperature_scaled"})
_FLOAT_COLUMNS = frozenset(
    [f"quality.{name}" for name in _QUALITY_SCALAR_FIELDS if name != "temperature_scaled"]
    + [f"robustness.{name}" for name in _ROBUSTNESS_FIELDS]
    + ["tok_s_per_gb"]
)


def _column_names() -> tuple[str, ...]:
    columns: list[str] = []
    columns += [f"provenance.{name}" for name in _PROVENANCE_FIELDS]
    columns += [f"backend.{name}" for name in _BACKEND_FIELDS]
    columns += ["variant_name", "family"]
    columns += [f"task.{name}" for name in _TASK_FIELDS]
    columns += [f"quality.{name}" for name in _QUALITY_SCALAR_FIELDS]
    columns.append("quality.ece_bin_sweep")
    columns += [f"robustness.{name}" for name in _ROBUSTNESS_FIELDS]
    columns += ["tok_s_per_gb", "latency", "memory"]
    return tuple(columns)


def _column_type(name: str) -> pa.DataType:
    if name in _INT_COLUMNS:
        return pa.int64()
    if name in _BOOL_COLUMNS:
        return pa.bool_()
    if name in _FLOAT_COLUMNS:
        return pa.float64()
    return pa.string()


RESULT_COLUMNS: tuple[str, ...] = _column_names()
RESULT_SCHEMA: pa.Schema = pa.schema([(name, _column_type(name)) for name in RESULT_COLUMNS])


def to_record(row: ResultRow) -> dict[str, Any]:
    """``ResultRow`` to a nested, JSON-safe dict (one jsonl line's content).

    ``dataclasses.asdict`` with one adjustment: ``ece_bin_sweep``'s int keys are
    stringified so the dict is JSON-native. Non-finite floats (``perplexity`` and
    ``tok_s_per_gb`` are ``NaN`` in WP3) are left as-is; the mirror is internal, so the
    default ``json`` ``NaN`` token is acceptable and round-trips through ``json.loads``.
    """
    record = dataclasses.asdict(row)
    sweep = record["quality"]["ece_bin_sweep"]
    record["quality"]["ece_bin_sweep"] = {str(key): value for key, value in sweep.items()}
    return record


def from_record(record: Mapping[str, Any]) -> ResultRow:
    """Nested dict to ``ResultRow``. Inverse of ``to_record``."""
    quality = dict(record["quality"])
    quality["ece_bin_sweep"] = {int(k): float(v) for k, v in quality["ece_bin_sweep"].items()}
    robustness = record["robustness"]
    return ResultRow(
        provenance=Provenance(**record["provenance"]),
        backend=Backend(**record["backend"]),
        variant_name=record["variant_name"],
        family=record["family"],
        task=TaskSpec(**record["task"]),
        quality=Quality(**quality),
        latency=[_to_latency(item) for item in record["latency"]],
        memory=[Memory(**item) for item in record["memory"]],
        tok_s_per_gb=record["tok_s_per_gb"],
        robustness=None if robustness is None else Robustness(**robustness),
    )


def _to_latency(item: Mapping[str, Any]) -> Latency:
    data = dict(item)
    data["machine_state"] = MachineState(**data["machine_state"])
    return Latency(**data)


def flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Nested record to a flat dotted dict keyed by ``RESULT_COLUMNS``."""
    flat: dict[str, Any] = {}
    for section, names in (
        ("provenance", _PROVENANCE_FIELDS),
        ("backend", _BACKEND_FIELDS),
        ("task", _TASK_FIELDS),
    ):
        block = record[section]
        for name in names:
            flat[f"{section}.{name}"] = block[name]
    quality = record["quality"]
    for name in _QUALITY_SCALAR_FIELDS:
        flat[f"quality.{name}"] = quality[name]
    flat["quality.ece_bin_sweep"] = json.dumps(quality["ece_bin_sweep"])
    flat["variant_name"] = record["variant_name"]
    flat["family"] = record["family"]
    flat["tok_s_per_gb"] = record["tok_s_per_gb"]
    robustness = record["robustness"]
    for name in _ROBUSTNESS_FIELDS:
        flat[f"robustness.{name}"] = None if robustness is None else robustness[name]
    flat["latency"] = json.dumps(record["latency"])
    flat["memory"] = json.dumps(record["memory"])
    return flat


def unflatten_record(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Flat dotted dict to a nested record. Inverse of ``flatten_record``."""
    quality = {name: flat[f"quality.{name}"] for name in _QUALITY_SCALAR_FIELDS}
    quality["ece_bin_sweep"] = json.loads(flat["quality.ece_bin_sweep"])
    robustness = (
        None
        if flat["robustness.permutation_consistency"] is None
        else {name: flat[f"robustness.{name}"] for name in _ROBUSTNESS_FIELDS}
    )
    return {
        "provenance": {name: flat[f"provenance.{name}"] for name in _PROVENANCE_FIELDS},
        "backend": {name: flat[f"backend.{name}"] for name in _BACKEND_FIELDS},
        "variant_name": flat["variant_name"],
        "family": flat["family"],
        "task": {name: flat[f"task.{name}"] for name in _TASK_FIELDS},
        "quality": quality,
        "latency": json.loads(flat["latency"]),
        "memory": json.loads(flat["memory"]),
        "tok_s_per_gb": flat["tok_s_per_gb"],
        "robustness": robustness,
    }
