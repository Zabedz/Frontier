"""Per-item ``(confidence, correct, gold, predicted)`` sidecars, one per appended result
row, at ``results/predictions/<key>.parquet``.

The row itself keeps only ECE scalars and a frozen bin-count sweep, so the reliability
gallery and the ECE-vs-bins sweep re-bin from these files. The key is derived from columns
already on the row, so the analysis join rebuilds it from the store frame alone.
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

    All four arrays have length ``n_items``.
    """

    confidence: FloatArray
    correct: CorrectArray
    gold: LabelArray
    predicted: LabelArray


def predictions_key(config_hash: str, seed: int, task_name: str) -> str:
    """Filename stem for a row's predictions sidecar.

    ``config_hash`` is taken over the fully merged config, which folds in the eval profile,
    so ``(config_hash, seed)`` already identifies the row; ``task_name`` keeps the stem
    readable and gives the analysis join a cross-check.
    """
    return f"{task_name}__{config_hash}__seed{seed}"


def predictions_path(root: Path, key: str) -> Path:
    """The sidecar path for ``key`` under ``root``."""
    return root / PREDICTIONS_SUBDIR / f"{key}.parquet"


def write_predictions_rows(rows: PredictionRows, *, root: Path, key: str) -> Path:
    """Write one sidecar atomically (tmp parquet, then ``Path.replace``); return its path.

    Raises ``ValueError``, naming each length, when the four arrays disagree.
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
