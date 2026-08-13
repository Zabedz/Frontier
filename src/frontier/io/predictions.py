"""Per-item ``(confidence, correct, gold, predicted)`` sidecars, one per appended result
row, at ``results/predictions/<key>.parquet``.

The row itself keeps only ECE scalars and a frozen bin-count sweep, so the reliability
gallery and the ECE-vs-bins sweep re-bin from these files. The key is derived from columns
already on the row, so the analysis join rebuilds it from the store frame alone.

Sidecars also carry the per-item distribution over answer letters, which post-hoc
recalibration needs. Probabilities are enough:
``log(p) = z - logZ``, and the constant cancels in the softmax, so a temperature fitted on
stored probabilities equals one fitted on the original logits. Sidecars written before that
column existed load with ``options=None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

# Redeclared rather than imported from metrics._array: that import executes
# metrics/__init__.py, which would pull scipy and schema into the io import path.
FloatArray = npt.NDArray[np.float64]
CorrectArray = npt.NDArray[np.bool_]
LabelArray = npt.NDArray[np.intp]
IntArray = npt.NDArray[np.intp]  # (n_items,), true option count per item
ProbMatrix = npt.NDArray[np.float64]  # (n_items, max_options), zero-padded
QidArray = npt.NDArray[np.str_]  # (n_items,), dataset item ids

PREDICTIONS_SUBDIR = "predictions"
PROBS_COLUMN = "probs"
QID_COLUMN = "qid"
SIMPLEX_ATOL = 1e-6

PREDICTIONS_SCHEMA: pa.Schema = pa.schema(
    [
        ("confidence", pa.float64()),
        ("correct", pa.bool_()),
        ("gold", pa.int64()),
        ("predicted", pa.int64()),
        (PROBS_COLUMN, pa.list_(pa.float64())),
        (QID_COLUMN, pa.string()),
    ]
)


class CorruptSidecarError(ValueError):
    """A sidecar whose contents contradict themselves. Always a defect, never an absence."""


class MissingSidecarError(ValueError):
    """No sidecar for a stored row. Expected for a store predating the sidecar."""


@dataclass(frozen=True, slots=True)
class OptionProbs:
    """Per-item distribution over answer letters.

    ``probs`` is zero-padded to the widest item, matching ``eval.extract.EvalOutputs``;
    ``n_options`` gives each item's true count, so columns at or past it are padding. The
    two travel together because a padded cell is a 0.0 whose ``log`` is ``-inf``.
    """

    probs: ProbMatrix  # (n_items, max_options)
    n_options: IntArray  # (n_items,)


@dataclass(frozen=True, slots=True)
class PredictionRows:
    """Per-item calibration signal for one scored row (one seed).

    Every array has length ``n_items``. ``options`` and ``qid`` are ``None`` for a sidecar
    written before those columns existed, and neither carries a default, so a producer
    stating them is a deliberate act.

    ``qid`` is the dataset's own item id. It cannot be recovered later without re-running,
    and the subject-level label-noise, raw-vs-redux, and answer-channel-agreement analyses
    all need to know which items they are looking at.
    """

    confidence: FloatArray
    correct: CorrectArray
    gold: LabelArray
    predicted: LabelArray
    options: OptionProbs | None
    qid: QidArray | None


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

    Each item's distribution is stored at its true option count, so the file holds no
    padding.

    Raises ``ValueError`` when the arrays disagree in length, when an option count is out of
    range, when a distribution does not sum to 1, or when the stored distribution disagrees
    with ``predicted`` or ``confidence``.
    """
    _check_lengths(rows)
    columns: dict[str, object] = {
        "confidence": np.asarray(rows.confidence, dtype=np.float64),
        "correct": np.asarray(rows.correct, dtype=np.bool_),
        "gold": np.asarray(rows.gold, dtype=np.int64),
        "predicted": np.asarray(rows.predicted, dtype=np.int64),
    }
    n_items = rows.confidence.shape[0]
    columns[QID_COLUMN] = [None] * n_items if rows.qid is None else rows.qid.tolist()
    if rows.options is None:
        columns[PROBS_COLUMN] = [None] * n_items
    else:
        _check_options(rows)
        counts = rows.options.n_options
        columns[PROBS_COLUMN] = [
            rows.options.probs[index, : counts[index]].astype(np.float64).tolist()
            for index in range(counts.shape[0])
        ]
    path = predictions_path(root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(columns, schema=PREDICTIONS_SCHEMA)
    tmp = path.with_name(f"{path.name}.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return path


def read_predictions(root: Path, key: str) -> PredictionRows:
    """Read the sidecar keyed by ``key`` under ``root``."""
    return read_predictions_path(predictions_path(root, key))


def read_predictions_path(path: Path) -> PredictionRows:
    """Read one sidecar parquet into a ``PredictionRows`` with exact dtypes.

    ``options`` is ``None`` when the file predates the distribution column or stores no
    distribution for any item.
    """
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
        options=_read_options(table),
        qid=_read_qid(table),
    )


def pad_option_probs(rows: list[FloatArray]) -> OptionProbs:
    """Pad ragged per-item distributions into one matrix plus their true counts."""
    counts = np.asarray([row.shape[0] for row in rows], dtype=np.intp)
    probs = np.zeros((len(rows), int(counts.max())), dtype=np.float64)
    for index, row in enumerate(rows):
        probs[index, : row.shape[0]] = row
    return OptionProbs(probs=probs, n_options=counts)


def _read_qid(table: pa.Table) -> QidArray | None:
    if QID_COLUMN not in table.column_names:
        return None
    stored = table.column(QID_COLUMN).to_pylist()
    missing = [index for index, item in enumerate(stored) if item is None]
    if len(missing) == len(stored):
        return None
    if missing:
        raise CorruptSidecarError(
            f"sidecar stores qids for some items but not {len(missing)} others (first at "
            f"index {missing[0]}); the writer is all-or-nothing, so this file is corrupt"
        )
    ids: QidArray = np.asarray(stored, dtype=np.str_)
    return ids


def _read_options(table: pa.Table) -> OptionProbs | None:
    if PROBS_COLUMN not in table.column_names:
        return None
    stored = table.column(PROBS_COLUMN).to_pylist()
    missing = [index for index, item in enumerate(stored) if item is None]
    if len(missing) == len(stored):
        return None
    if missing:
        raise CorruptSidecarError(
            f"sidecar stores distributions for some items but not {len(missing)} others "
            f"(first at index {missing[0]}); the writer is all-or-nothing, so this file is "
            f"corrupt"
        )
    return pad_option_probs([np.asarray(item, dtype=np.float64) for item in stored])


def _check_lengths(rows: PredictionRows) -> None:
    lengths = {
        "confidence": rows.confidence.shape[0],
        "correct": rows.correct.shape[0],
        "gold": rows.gold.shape[0],
        "predicted": rows.predicted.shape[0],
    }
    if rows.options is not None:
        lengths["probs"] = rows.options.probs.shape[0]
        lengths["n_options"] = rows.options.n_options.shape[0]
    if rows.qid is not None:
        lengths["qid"] = rows.qid.shape[0]
    if len(set(lengths.values())) != 1:
        raise ValueError(f"prediction arrays disagree in length: {lengths}")


def _check_options(rows: PredictionRows) -> None:
    """Reject a distribution that contradicts the scalars stored beside it.

    A sidecar whose vector disagrees with its own ``predicted`` or ``confidence`` would
    recalibrate to a plausible wrong number with nothing to flag it.
    """
    assert rows.options is not None  # narrowed by the caller
    probs = rows.options.probs
    counts = rows.options.n_options
    width = probs.shape[1]
    bad = np.flatnonzero((counts < 1) | (counts > width))
    if bad.size:
        first = int(bad[0])
        raise ValueError(f"item {first} has n_options={int(counts[first])}, outside [1, {width}]")
    # NaN fails every comparison below, and an all-NaN row's argmax is 0, which agrees with
    # a predicted also taken as that row's argmax. A -inf logit for a letter the backend
    # never returned reaches here as NaN, so the finite check goes first.
    live = np.arange(width)[None, :] < counts[:, None]
    if not bool(np.isfinite(probs[live]).all()) or not bool(np.isfinite(rows.confidence).all()):
        item = int(np.argmax(~np.isfinite(np.where(live, probs, 0.0)).all(axis=1)))
        raise ValueError(f"item {item} distribution is not finite: {probs[item]}")
    for index in range(counts.shape[0]):
        row = probs[index, : counts[index]]
        total = float(row.sum())
        if abs(total - 1.0) > SIMPLEX_ATOL:
            raise ValueError(f"item {index} distribution sums to {total}, expected 1.0")
        if int(row.argmax()) != int(rows.predicted[index]):
            raise ValueError(
                f"item {index} distribution peaks at option {int(row.argmax())} but "
                f"predicted is {int(rows.predicted[index])}"
            )
        if abs(float(row.max()) - float(rows.confidence[index])) > SIMPLEX_ATOL:
            raise ValueError(
                f"item {index} distribution peaks at {float(row.max())} but confidence is "
                f"{float(rows.confidence[index])}"
            )
