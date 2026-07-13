"""load_tidy / collapse_seeds over a store built from the io row fixtures.

The io fixtures live one directory over (``tests/io/rows.py``); pytest puts each test
file's own directory on ``sys.path``, so the shared fixture module is reached by adding
that sibling directory here, the same ``from rows import`` the io tests use.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from frontier.analysis.load import (
    collapse_seeds,
    load_all_predictions,
    load_tidy,
    prediction_labels,
)
from frontier.io.predictions import PredictionRows, predictions_key, write_predictions_rows
from frontier.io.store import ResultStore, append_row
from frontier.schema import ResultRow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))

from rows import row_with_latency, sample_row

SEEDS = (0, 1, 2)
FP16_ITL_MS = 3.1
FP16_PEAK_VRAM_MB = 8100.0
FP16_MMLU_HASH = "0" * 64  # sample_row's config_hash
FP16_ARC_HASH = "3" * 64
TWO_TASK_LABELS = {"fp16 / mmlu", "fp16 / arc_challenge"}


def _preds(n: int = 40) -> PredictionRows:
    rng = np.random.default_rng(0)
    confidence = rng.uniform(0.3, 1.0, size=n)
    correct = rng.uniform(0.0, 1.0, size=n) < confidence
    gold = rng.integers(0, 4, size=n).astype(np.intp)
    predicted = np.where(correct, gold, (gold + 1) % 4).astype(np.intp)
    return PredictionRows(confidence, correct.astype(np.bool_), gold, predicted)


def _variant(
    base: ResultRow,
    *,
    name: str,
    family: str,
    track: str,
    config_hash: str,
    task_name: str = "mmlu",
) -> ResultRow:
    return replace(
        base,
        variant_name=name,
        family=family,  # type: ignore[arg-type]
        backend=replace(base.backend, track=track),  # type: ignore[arg-type]
        provenance=replace(base.provenance, config_hash=config_hash),
        task=replace(base.task, task_name=task_name),
    )


def test_load_tidy_parses_identity_and_cost_columns(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    append_row(row_with_latency(), store)  # fp16, baseline, populated latency/memory
    append_row(
        _variant(sample_row(), name="int4-nf4", family="ptq", track="A", config_hash="1" * 64),
        store,
    )

    tidy = load_tidy(store)
    by_variant = {record["variant_name"]: record for record in tidy.to_dict("records")}

    fp16 = by_variant["fp16"]
    assert fp16["family"] == "baseline"
    assert fp16["track"] == "A"
    assert fp16["accuracy"] == pytest.approx(0.82)
    assert fp16["itl_median_ms"] == pytest.approx(FP16_ITL_MS)
    assert fp16["peak_vram_mb"] == pytest.approx(FP16_PEAK_VRAM_MB)

    nf4 = by_variant["int4-nf4"]
    assert nf4["family"] == "ptq"
    # sample_row carries no latency/memory, so its cost columns are NaN (dropped later).
    assert math.isnan(nf4["itl_median_ms"])
    assert math.isnan(nf4["peak_vram_mb"])


def test_collapse_seeds_reduces_three_seeds_to_one_row(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    for seed in SEEDS:
        append_row(sample_row(seed=seed), store)

    collapsed = collapse_seeds(load_tidy(store))
    assert len(collapsed) == 1
    row = collapsed.to_dict("records")[0]
    assert row["variant_name"] == "fp16"
    assert row["n_seeds"] == len(SEEDS)
    assert row["accuracy"] == pytest.approx(0.82)


def test_task_name_filter_selects_one_task(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    append_row(sample_row(), store)  # task_name mmlu
    append_row(
        _variant(
            sample_row(),
            name="fp16",
            family="baseline",
            track="A",
            config_hash="2" * 64,
            task_name="arc_challenge",
        ),
        store,
    )

    mmlu = load_tidy(store, task_name="mmlu")
    assert set(mmlu["task_name"].tolist()) == {"mmlu"}
    arc = load_tidy(store, task_name="arc_challenge")
    assert set(arc["task_name"].tolist()) == {"arc_challenge"}


def test_load_all_predictions_keeps_two_tasks_of_one_variant(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    append_row(sample_row(), store)  # fp16 on mmlu, config_hash FP16_MMLU_HASH
    append_row(
        _variant(
            sample_row(),
            name="fp16",
            family="baseline",
            track="A",
            config_hash=FP16_ARC_HASH,
            task_name="arc_challenge",
        ),
        store,
    )
    write_predictions_rows(_preds(), root=tmp_path, key=predictions_key(FP16_MMLU_HASH, 0, "mmlu"))
    write_predictions_rows(
        _preds(), root=tmp_path, key=predictions_key(FP16_ARC_HASH, 0, "arc_challenge")
    )

    tidy = load_tidy(store)  # both tasks present
    pooled = load_all_predictions(tidy, root=tmp_path)
    # One variant_name on two tasks must not collide: composite labels keep both.
    assert len(pooled) == len(TWO_TASK_LABELS)
    assert set(pooled) == TWO_TASK_LABELS
    assert set(prediction_labels(tidy)) == TWO_TASK_LABELS
