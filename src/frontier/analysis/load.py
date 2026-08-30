"""Read the result store into a tidy per-variant frame and join the prediction sidecars.

A sidecar is found by rebuilding its key from the row's ``config_hash``, ``seed``, and
``task_name``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from frontier.analysis.holdout import split as split_rows
from frontier.io.predictions import (
    MissingSidecarError,
    OptionProbs,
    PredictionRows,
    ProbMatrix,
    QidArray,
    predictions_key,
    predictions_path,
    read_predictions_path,
)
from frontier.io.store import ResultStore, read_frame


class MixedConfigHashError(ValueError):
    """One variant's rows scored under more than one config hash, so they cannot pool."""


XCost = Literal["latency", "memory", "cost_inv"]

COST_INV_TOKENS = 1000.0


@dataclass(frozen=True, slots=True)
class XAxisSpec:
    """A cost axis for the frontier chart. All three minimize (lower is better).

    ``cross_track``: comparable across the two inference tracks (the fairness rule in
    ``docs/architecture.md``). Memory qualifies; latency and the throughput-derived proxy
    embed a per-backend clock, so one column of them mixes llama.cpp with vLLM/HF.
    """

    key: XCost
    column: str
    label: str
    cross_track: bool


X_AXES: dict[XCost, XAxisSpec] = {
    "latency": XAxisSpec(
        "latency", "itl_median_ms", "Inter-token latency (ms, decode)", cross_track=False
    ),
    "memory": XAxisSpec("memory", "peak_vram_mb", "Peak VRAM (MB)", cross_track=True),
    "cost_inv": XAxisSpec("cost_inv", "cost_inv", "GB-seconds per 1000 tokens", cross_track=False),
}

TIDY_COLUMNS: tuple[str, ...] = (
    "variant_name",
    "family",
    "track",
    "backend",
    "task_name",
    "config_hash",
    "seed",
    "accuracy",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "ece_equal_width",
    "ece_ci_low",
    "ece_ci_high",
    "ece_equal_mass_ace",
    "brier_reliability",
    "tok_s_per_gb",
    "itl_median_ms",
    "ttft_median_ms",
    "throughput_tok_s",
    "peak_vram_mb",
    "cost_inv",
)

_NUMERIC_COLUMNS: tuple[str, ...] = (
    "accuracy",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "ece_equal_width",
    "ece_ci_low",
    "ece_ci_high",
    "ece_equal_mass_ace",
    "brier_reliability",
    "tok_s_per_gb",
    "itl_median_ms",
    "ttft_median_ms",
    "throughput_tok_s",
    "peak_vram_mb",
    "cost_inv",
)


@dataclass(frozen=True, slots=True)
class _LatencyPick:
    itl: float
    ttft: float
    throughput: float


def load_tidy(
    store: ResultStore,
    *,
    task_name: str | None = None,
    batch_size: int = 1,
    context_len: int | None = None,
) -> pd.DataFrame:
    """Read the store into one tidy row per stored ``ResultRow`` (per seed).

    ``context_len`` of ``None`` picks the smallest stored context, ``task_name`` of
    ``None`` keeps every task. ``batch_size`` picks the stored latency and memory entry; a
    size the run never profiled leaves those columns ``NaN`` too, as does a skipped
    profile, and the Pareto step drops them. ``cost_inv`` is ``1000 / tok_s_per_gb``, so it
    minimizes like the other cost axes.
    """
    frame = read_frame(store)
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        if task_name is not None and record["task.task_name"] != task_name:
            continue
        latency = _pick_latency(json.loads(record["latency"]), batch_size)
        memory = _pick_memory(json.loads(record["memory"]), batch_size, context_len)
        tok_s_per_gb = float(record["tok_s_per_gb"])
        rows.append(
            {
                "variant_name": record["variant_name"],
                "family": record["family"],
                "track": record["backend.track"],
                "backend": record["backend.inference_backend"],
                "task_name": record["task.task_name"],
                "config_hash": record["provenance.config_hash"],
                "seed": int(record["provenance.seed"]),
                "accuracy": float(record["quality.accuracy"]),
                "accuracy_ci_low": float(record["quality.accuracy_ci_low"]),
                "accuracy_ci_high": float(record["quality.accuracy_ci_high"]),
                "ece_equal_width": float(record["quality.ece_equal_width"]),
                "ece_ci_low": float(record["quality.ece_ci_low"]),
                "ece_ci_high": float(record["quality.ece_ci_high"]),
                "ece_equal_mass_ace": float(record["quality.ece_equal_mass_ace"]),
                "brier_reliability": float(record["quality.brier_reliability"]),
                "tok_s_per_gb": tok_s_per_gb,
                "itl_median_ms": latency.itl,
                "ttft_median_ms": latency.ttft,
                "throughput_tok_s": latency.throughput,
                "peak_vram_mb": memory,
                "cost_inv": _cost_inv(tok_s_per_gb),
            }
        )
    return pd.DataFrame.from_records(rows, columns=list(TIDY_COLUMNS))


def collapse_seeds(tidy: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(variant_name, task_name)``: mean the numeric columns across seeds.

    The CI columns average the per-seed intervals, which is a display interval; a
    multi-seed CI (seed variance plus the bootstrap) waits for real multi-seed rows.
    """
    if tidy.empty:
        empty = tidy.copy()
        empty["n_seeds"] = pd.Series(dtype="int64")
        return empty
    grouped = tidy.groupby(["variant_name", "task_name"], sort=False)
    means = grouped[list(_NUMERIC_COLUMNS)].mean()
    carried = grouped[["family", "track", "backend", "config_hash"]].first()
    n_seeds = grouped.size().rename("n_seeds")
    collapsed = pd.concat([carried, means, n_seeds], axis=1).reset_index()
    return collapsed


def load_predictions_for_variant(
    tidy: pd.DataFrame, *, variant_name: str, task_name: str, root: Path
) -> PredictionRows:
    """Pool every matching seed's sidecar for one variant into one ``PredictionRows``.

    Concatenation is order-independent for ECE and reliability, so the seeds need no
    weighting. A group spanning more than one ``config_hash`` is refused: pooling runs
    scored differently would raise the item count while the paired guards downstream
    stayed satisfied, a doubled group matching item for item against another doubled one.

    Raises ``ValueError`` for a missing variant/task, ``MixedConfigHashError`` for a group
    spanning several config hashes, and ``MissingSidecarError`` for a missing sidecar.
    """
    return _concat_predictions(_sidecars_for_variant(tidy, variant_name, task_name, root))


def load_split_predictions(
    tidy: pd.DataFrame, *, variant_name: str, task_name: str, root: Path
) -> tuple[PredictionRows, PredictionRows]:
    """One variant's sidecars pooled into ``(fit, report)`` halves.

    The split is keyed on each item's id, so an item lands on the same side whichever seed
    or subset it arrived in, and pooling order stops mattering. See ``holdout``.
    """
    pieces = _sidecars_for_variant(tidy, variant_name, task_name, root)
    halves = [split_rows(piece) for piece in pieces]
    return (
        _concat_predictions([fit for fit, _report in halves]),
        _concat_predictions([report for _fit, report in halves]),
    )


def _sidecars_for_variant(
    tidy: pd.DataFrame, variant_name: str, task_name: str, root: Path
) -> list[PredictionRows]:
    subset = tidy[(tidy["variant_name"] == variant_name) & (tidy["task_name"] == task_name)]
    if subset.empty:
        raise ValueError(
            f"no rows in the tidy frame for variant {variant_name!r} on task {task_name!r}"
        )
    hashes = sorted({str(value) for value in subset["config_hash"]})
    if len(hashes) > 1:
        raise MixedConfigHashError(
            f"variant {variant_name!r} on task {task_name!r} spans {len(hashes)} config "
            f"hashes ({', '.join(hashes)}); pooling them would mix runs scored differently"
        )
    pieces = _collect_sidecars(subset, task_name, root)
    if not pieces:
        raise MissingSidecarError(
            f"no predictions sidecar found under {root} for variant {variant_name!r} "
            f"on task {task_name!r}"
        )
    return pieces


def load_all_predictions(tidy: pd.DataFrame, *, root: Path) -> dict[str, PredictionRows]:
    """Map a variant label to its pooled ``PredictionRows`` for the gallery and sweep.

    The label takes a ``" / task"`` suffix once the frame spans more than one task, so a
    variant scored on two tasks keeps two entries. Variants whose sidecars are absent (a
    store predating the sidecar) are skipped, so the figures degrade gracefully.
    """
    multi = _spans_multiple_tasks(tidy)
    result: dict[str, PredictionRows] = {}
    for _, subset in tidy.groupby(["variant_name", "task_name"], sort=False):
        variant_name = str(subset["variant_name"].iloc[0])
        task_name = str(subset["task_name"].iloc[0])
        pieces = _collect_sidecars(subset, task_name, root)
        if pieces:
            result[_group_label(variant_name, task_name, multi=multi)] = _concat_predictions(pieces)
    return result


def prediction_labels(tidy: pd.DataFrame) -> list[str]:
    """Every label ``load_all_predictions`` would emit if all sidecars were present.

    The caller diffs this against that mapping to name the variants dropped for want of
    a sidecar.
    """
    multi = _spans_multiple_tasks(tidy)
    labels: list[str] = []
    for _, subset in tidy.groupby(["variant_name", "task_name"], sort=False):
        variant_name = str(subset["variant_name"].iloc[0])
        task_name = str(subset["task_name"].iloc[0])
        labels.append(_group_label(variant_name, task_name, multi=multi))
    return labels


def _spans_multiple_tasks(tidy: pd.DataFrame) -> bool:
    return len(tidy["task_name"].unique()) > 1


def _group_label(variant_name: str, task_name: str, *, multi: bool) -> str:
    return f"{variant_name} / {task_name}" if multi else variant_name


def _collect_sidecars(subset: pd.DataFrame, task_name: str, root: Path) -> list[PredictionRows]:
    pieces: list[PredictionRows] = []
    for record in subset.to_dict("records"):
        key = predictions_key(str(record["config_hash"]), int(record["seed"]), task_name)
        path = predictions_path(root, key)
        if path.exists():
            pieces.append(read_predictions_path(path))
    return pieces


def _concat_predictions(pieces: list[PredictionRows]) -> PredictionRows:
    return PredictionRows(
        confidence=np.concatenate([piece.confidence for piece in pieces]),
        correct=np.concatenate([piece.correct for piece in pieces]),
        gold=np.concatenate([piece.gold for piece in pieces]),
        predicted=np.concatenate([piece.predicted for piece in pieces]),
        options=_concat_options(pieces),
        qid=_concat_qid(pieces),
    )


def _concat_qid(pieces: list[PredictionRows]) -> QidArray | None:
    """One seed missing its ids drops the pool, on the same reasoning as the options."""
    seen: list[QidArray] = []
    for piece in pieces:
        if piece.qid is None:
            return None
        seen.append(piece.qid)
    pooled: QidArray = np.concatenate(seen)
    return pooled


def _concat_options(pieces: list[PredictionRows]) -> OptionProbs | None:
    """Pool the per-item distributions, re-padding to the widest seed.

    One seed missing its distribution drops the whole pool to ``None``; a matrix covering
    part of the items would recalibrate on a subset with nothing to signal it.
    """
    seen: list[OptionProbs] = []
    for piece in pieces:
        if piece.options is None:
            return None
        seen.append(piece.options)
    width = max(item.probs.shape[1] for item in seen)
    padded: list[ProbMatrix] = []
    for item in seen:
        if item.probs.shape[1] == width:
            padded.append(item.probs)
            continue
        grown = np.zeros((item.probs.shape[0], width), dtype=np.float64)
        grown[:, : item.probs.shape[1]] = item.probs
        padded.append(grown)
    return OptionProbs(
        probs=np.concatenate(padded),
        n_options=np.concatenate([item.n_options for item in seen]),
    )


def _pick_latency(entries: list[dict[str, Any]], batch_size: int) -> _LatencyPick:
    for entry in entries:
        if entry["batch_size"] == batch_size:
            return _LatencyPick(
                itl=float(entry["itl_median_ms"]),
                ttft=float(entry["ttft_median_ms"]),
                throughput=float(entry["throughput_tok_s"]),
            )
    return _LatencyPick(itl=math.nan, ttft=math.nan, throughput=math.nan)


def _pick_memory(entries: list[dict[str, Any]], batch_size: int, context_len: int | None) -> float:
    matching = [entry for entry in entries if entry["batch_size"] == batch_size]
    if not matching:
        return math.nan
    if context_len is None:
        smallest = min(matching, key=lambda entry: entry["context_len"])
        return float(smallest["peak_vram_mb"])
    for entry in matching:
        if entry["context_len"] == context_len:
            return float(entry["peak_vram_mb"])
    return math.nan


def _cost_inv(tok_s_per_gb: float) -> float:
    if not math.isfinite(tok_s_per_gb) or tok_s_per_gb <= 0.0:
        return math.nan
    return COST_INV_TOKENS / tok_s_per_gb
