"""The calibration-set axis (WP6): each out-of-domain variant against its in-domain twin,
and the pre-registered contrast between two methods' shifts.

Every statistic is paired on the items, so twins are checked item for item first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from frontier.analysis._skipped import Skipped
from frontier.analysis.load import load_predictions_for_variant
from frontier.io.predictions import PredictionRows
from frontier.metrics.bootstrap import (
    DEFAULT_RESAMPLES,
    ConfidenceInterval,
    paired_delta_accuracy_ci,
    paired_delta_confidence_ci,
    paired_delta_ece_ci,
    paired_shift_difference_ci,
)
from frontier.metrics.calibration import DEFAULT_BINS

DEFAULT_PLAN_PATH = Path("configs/analysis/calibration_set.yaml")


@dataclass(frozen=True, slots=True)
class CalibrationSetPlan:
    """``twins`` maps each out-of-domain variant to its in-domain twin. ``contrast`` names
    two out-of-domain variants whose shifts are differenced, first minus second."""

    twins: dict[str, str]
    contrast: tuple[str, str] | None


@dataclass(frozen=True, slots=True)
class CorpusShift:
    """One method's change when its calibration corpus moves out of domain.

    Every delta is out-of-domain minus in-domain. ``answer_agreement`` is the share of
    items both rows answer with the same letter.
    """

    variant: str
    twin: str
    task: str
    n_items: int
    delta_accuracy: ConfidenceInterval
    delta_ece: ConfidenceInterval
    delta_confidence: ConfidenceInterval
    answer_agreement: float


@dataclass(frozen=True, slots=True)
class ShiftContrast:
    """``first``'s corpus shift minus ``second``'s, all four rows on the same items."""

    first: str
    second: str
    task: str
    n_items: int
    ece: ConfidenceInterval
    confidence: ConfidenceInterval


def load_plan(path: Path = DEFAULT_PLAN_PATH) -> CalibrationSetPlan:
    """Read the twin map and the optional contrast."""
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("pairs"), dict):
        raise ValueError(f"{path} has no top-level 'pairs' mapping")
    twins = {str(variant): str(twin) for variant, twin in loaded["pairs"].items()}
    contrast_block = loaded.get("contrast")
    if contrast_block is None:
        return CalibrationSetPlan(twins=twins, contrast=None)
    if not isinstance(contrast_block, dict) or {"first", "second"} - set(contrast_block):
        raise ValueError(f"{path}: 'contrast' needs 'first' and 'second'")
    contrast = (str(contrast_block["first"]), str(contrast_block["second"]))
    unknown = [name for name in contrast if name not in twins]
    if unknown:
        raise ValueError(f"{path}: contrast names {unknown} have no entry under 'pairs'")
    return CalibrationSetPlan(twins=twins, contrast=contrast)


def corpus_shift(
    variant: str,
    twin: str,
    task: str,
    out_rows: PredictionRows,
    in_rows: PredictionRows,
    *,
    n_bins: int = DEFAULT_BINS,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: int | None = 0,
) -> CorpusShift:
    """The paired deltas, out-of-domain minus in-domain, on item-aligned rows."""
    return CorpusShift(
        variant=variant,
        twin=twin,
        task=task,
        n_items=int(out_rows.gold.shape[0]),
        delta_accuracy=paired_delta_accuracy_ci(
            in_rows.correct, out_rows.correct, n_resamples=n_resamples, rng=rng
        ),
        delta_ece=paired_delta_ece_ci(
            in_rows.confidence,
            in_rows.correct,
            out_rows.confidence,
            out_rows.correct,
            n_bins=n_bins,
            n_resamples=n_resamples,
            rng=rng,
        ),
        delta_confidence=paired_delta_confidence_ci(
            in_rows.confidence, out_rows.confidence, n_resamples=n_resamples, rng=rng
        ),
        answer_agreement=float(np.mean(out_rows.predicted == in_rows.predicted)),
    )


def shift_contrast(
    first: str,
    second: str,
    task: str,
    rows: Sequence[PredictionRows],
    *,
    n_bins: int = DEFAULT_BINS,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: int | None = 0,
) -> ShiftContrast:
    """``rows`` ordered first's twin, first, second's twin, second, all item-aligned."""
    confidence = [row.confidence for row in rows]
    correct = [row.correct for row in rows]
    return ShiftContrast(
        first=first,
        second=second,
        task=task,
        n_items=int(rows[0].gold.shape[0]),
        ece=paired_shift_difference_ci(
            confidence, correct, metric="ece", n_bins=n_bins, n_resamples=n_resamples, rng=rng
        ),
        confidence=paired_shift_difference_ci(
            confidence, correct, metric="confidence", n_resamples=n_resamples, rng=rng
        ),
    )


def calibration_set_table(
    tidy: pd.DataFrame,
    *,
    root: Path,
    plan: CalibrationSetPlan,
    n_bins: int = DEFAULT_BINS,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: int | None = 0,
) -> tuple[list[CorpusShift], list[ShiftContrast], list[Skipped]]:
    """Every twin pair and the contrast, on each task separately.

    A variant with no row, a missing sidecar, or rows under two config hashes is skipped
    with that reason. Twins whose sidecars describe different items raise, since the axis
    holds everything but the corpus fixed.
    """
    shifts: list[CorpusShift] = []
    contrasts: list[ShiftContrast] = []
    skipped: list[Skipped] = []
    if tidy.empty:
        return shifts, contrasts, skipped
    for task in sorted(str(name) for name in tidy["task_name"].unique()):
        task_rows = _TaskRows(tidy, root=root, task=task, skipped=skipped)
        for variant, twin in plan.twins.items():
            out_rows = task_rows.get(variant, owner=variant)
            in_rows = task_rows.get(twin, owner=variant)
            if out_rows is None or in_rows is None:
                continue
            _require_same_items(variant, out_rows, twin, in_rows, task)
            shifts.append(
                corpus_shift(
                    variant,
                    twin,
                    task,
                    out_rows,
                    in_rows,
                    n_bins=n_bins,
                    n_resamples=n_resamples,
                    rng=rng,
                )
            )
        if plan.contrast is not None:
            first, second = plan.contrast
            names = (plan.twins[first], first, plan.twins[second], second)
            label = f"{first} - {second}"
            rows = [task_rows.get(name, owner=label) for name in names]
            if any(row is None for row in rows):
                continue
            ready = [row for row in rows if row is not None]
            for name, row in zip(names[1:], ready[1:], strict=True):
                _require_same_items(name, row, names[0], ready[0], task)
            contrasts.append(
                shift_contrast(
                    first, second, task, ready, n_bins=n_bins, n_resamples=n_resamples, rng=rng
                )
            )
    return shifts, contrasts, skipped


def shifts_to_frame(shifts: Sequence[CorpusShift]) -> pd.DataFrame:
    """One row per twin pair."""
    records = []
    for shift in shifts:
        record: dict[str, object] = {
            "variant": shift.variant,
            "twin": shift.twin,
            "task": shift.task,
            "n_items": shift.n_items,
            "answer_agreement": shift.answer_agreement,
        }
        for name, interval in (
            ("delta_accuracy", shift.delta_accuracy),
            ("delta_ece", shift.delta_ece),
            ("delta_confidence", shift.delta_confidence),
        ):
            record |= _interval_columns(name, interval)
        records.append(record)
    return pd.DataFrame.from_records(records)


def contrasts_to_frame(contrasts: Sequence[ShiftContrast]) -> pd.DataFrame:
    """One row per contrast."""
    records = []
    for contrast in contrasts:
        record: dict[str, object] = {
            "first": contrast.first,
            "second": contrast.second,
            "task": contrast.task,
            "n_items": contrast.n_items,
        }
        record |= _interval_columns("ece_shift_difference", contrast.ece)
        record |= _interval_columns("confidence_shift_difference", contrast.confidence)
        records.append(record)
    return pd.DataFrame.from_records(records)


class _TaskRows:
    """One task's sidecars, loaded once each, with the reason any variant could not load."""

    def __init__(self, tidy: pd.DataFrame, *, root: Path, task: str, skipped: list[Skipped]):
        self._tidy = tidy
        self._root = root
        self._task = task
        self._skipped = skipped
        self._present = {str(name) for name in tidy.loc[tidy["task_name"] == task, "variant_name"]}
        self._loaded: dict[str, PredictionRows] = {}

    def get(self, name: str, *, owner: str) -> PredictionRows | None:
        if name in self._loaded:
            return self._loaded[name]
        if name not in self._present:
            self._skipped.append(
                Skipped(owner, self._task, f"{name} has no row on task {self._task}")
            )
            return None
        try:
            rows = load_predictions_for_variant(
                self._tidy, variant_name=name, task_name=self._task, root=self._root
            )
        except ValueError as unusable:
            self._skipped.append(Skipped(owner, self._task, f"{name}: {unusable}"))
            return None
        self._loaded[name] = rows
        return rows


def _interval_columns(name: str, interval: ConfidenceInterval) -> dict[str, object]:
    return {
        name: interval.point,
        f"{name}_low": interval.low,
        f"{name}_high": interval.high,
        f"{name}_excludes_zero": interval.excludes_zero,
    }


def _require_same_items(
    name: str, rows: PredictionRows, twin: str, twin_rows: PredictionRows, task: str
) -> None:
    if rows.gold.shape != twin_rows.gold.shape:
        raise ValueError(
            f"cannot pair {name} with {twin} on {task}: sidecars hold {rows.gold.shape[0]} and "
            f"{twin_rows.gold.shape[0]} items"
        )
    if rows.qid is not None and twin_rows.qid is not None:
        if not np.array_equal(rows.qid, twin_rows.qid):
            first = int(np.flatnonzero(rows.qid != twin_rows.qid)[0])
            raise ValueError(
                f"cannot pair {name} with {twin} on {task}: item {first} is "
                f"{rows.qid[first]!r} in one and {twin_rows.qid[first]!r} in the other"
            )
        return
    if not np.array_equal(rows.gold, twin_rows.gold):
        first = int(np.flatnonzero(rows.gold != twin_rows.gold)[0])
        raise ValueError(
            f"cannot pair {name} with {twin} on {task}: gold differs first at index {first}"
        )
