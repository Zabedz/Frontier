"""Per-run predictions sidecar: the raw calibration signal behind each result row.

The aggregate ``ResultRow`` keeps only ECE scalars and a frozen bin-count sweep; it
retains no per-item confidence. The reliability gallery and the ECE-vs-bins sweep need
the per-item ``(confidence, correct)`` arrays to re-bin freely, so the runner writes
them alongside each appended row as ``results/predictions/<key>.parquet``. Analysis
reads the store for the frontier chart and these sidecars for the calibration figures.

The key is derived from columns already on the row (config hash, seed, task name), so
the analysis join reconstructs it from the store frame with no separate index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

FloatArray = npt.NDArray[np.float64]
CorrectArray = npt.NDArray[np.bool_]
LabelArray = npt.NDArray[np.intp]

PREDICTIONS_SUBDIR = "predictions"

PREDICTIONS_SCHEMA: pa.Schema = pa.schema(
    [
        ("confidence", pa.float64()),
        ("correct", pa.bool_()),
        ("gold", pa.int64()),
        ("predicted", pa.int64()),
    ]
)


@dataclass(frozen=True, slots=True)
class PredictionRows:
    """Per-item calibration signal for one scored row (one seed).

    ``confidence`` and ``correct`` are the load-bearing pair for re-binning ECE and
    drawing reliability; ``gold`` and ``predicted`` make the file self-describing and
    let an independent accuracy check run off the sidecar. All four are the same
    length ``n_items``.
    """

    confidence: FloatArray
    correct: CorrectArray
    gold: LabelArray
    predicted: LabelArray


def predictions_key(config_hash: str, seed: int, task_name: str) -> str:
    """Filename stem for a row's predictions sidecar.

    A row is uniquely identified by ``(config_hash, seed)``: ``config_hash`` is taken
    over the fully merged config, which already folds in the eval profile, so two
    profiles of the same variant get different hashes. ``task_name`` is folded in only
    to keep the stem readable and to give the analysis join a defensive cross-check.
    All three are columns on the stored row, so the key is reconstructable from the
    store frame alone.
    """
    return f"{task_name}__{config_hash}__seed{seed}"


def predictions_path(root: Path, key: str) -> Path:
    """The sidecar path ``root / PREDICTIONS_SUBDIR / f"{key}.parquet"``."""
    return root / PREDICTIONS_SUBDIR / f"{key}.parquet"


def write_predictions_rows(rows: PredictionRows, *, root: Path, key: str) -> Path:
    """Write one sidecar atomically (tmp parquet, then ``Path.replace``); return its path.

    Raises
    ------
    ValueError
        If the four arrays disagree in length. The message names each length.
    """
    _check_lengths(rows)
    path = predictions_path(root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(
        {
            "confidence": np.asarray(rows.confidence, dtype=np.float64),
            "correct": np.asarray(rows.correct, dtype=np.bool_),
            "gold": np.asarray(rows.gold, dtype=np.int64),
            "predicted": np.asarray(rows.predicted, dtype=np.int64),
        },
        schema=PREDICTIONS_SCHEMA,
    )
    tmp = path.with_name(f"{path.name}.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return path


def read_predictions(root: Path, key: str) -> PredictionRows:
    """Read the sidecar keyed by ``key`` under ``root``."""
    return read_predictions_path(predictions_path(root, key))


def read_predictions_path(path: Path) -> PredictionRows:
    """Read one sidecar parquet into a ``PredictionRows`` with exact dtypes."""
    table = pq.read_table(path)
    return PredictionRows(
        confidence=np.asarray(
            table.column("confidence").to_numpy(zero_copy_only=False), dtype=np.float64
        ),
        correct=np.asarray(table.column("correct").to_numpy(zero_copy_only=False), dtype=np.bool_),
        gold=np.asarray(table.column("gold").to_numpy(zero_copy_only=False), dtype=np.intp),
        predicted=np.asarray(
            table.column("predicted").to_numpy(zero_copy_only=False), dtype=np.intp
        ),
    )


def _check_lengths(rows: PredictionRows) -> None:
    lengths = {
        "confidence": rows.confidence.shape[0],
        "correct": rows.correct.shape[0],
        "gold": rows.gold.shape[0],
        "predicted": rows.predicted.shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"prediction arrays disagree in length: {lengths}")
