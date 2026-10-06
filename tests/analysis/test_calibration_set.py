"""The calibration-set axis over a seeded store: twin shifts, the contrast, and the command."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from frontier.analysis import load_tidy
from frontier.analysis._skipped import Skipped
from frontier.analysis.calibration_set import (
    DEFAULT_PLAN_PATH,
    CalibrationSetPlan,
    CorpusShift,
    ShiftContrast,
    calibration_set_table,
    load_plan,
)
from frontier.io.predictions import PredictionRows, predictions_key, write_predictions_rows
from frontier.io.store import ResultStore, append_row
from frontier.pipeline.cli import app

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))

from rows import sample_row

REPO_PLAN = Path(__file__).resolve().parents[2] / DEFAULT_PLAN_PATH
N_ITEMS = 300
SHIFT = 0.05
RESAMPLES = 19
ACCURACY = 0.6
PLAN = CalibrationSetPlan(
    twins={"int4-gptq-ood": "int4-gptq", "int4-awq-ood": "int4-awq"},
    contrast=("int4-awq-ood", "int4-gptq-ood"),
)
# AWQ's out-of-domain row is SHIFT more confident than its twin; GPTQ's is unchanged.
CONFIDENCE_SHIFT = {"int4-gptq": 0.0, "int4-gptq-ood": 0.0, "int4-awq": 0.0, "int4-awq-ood": SHIFT}


def _seed(
    root: Path, variants: dict[str, float], *, qid_prefix: dict[str, str] | None = None
) -> None:
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 4, N_ITEMS).astype(np.intp)
    predicted = np.where(rng.random(N_ITEMS) < ACCURACY, gold, (gold + 1) % 4).astype(np.intp)
    base_confidence = rng.uniform(0.5, 0.9, N_ITEMS)
    store = ResultStore(root)
    for index, (name, shift) in enumerate(variants.items()):
        config_hash = f"{index:x}" * 64
        row = sample_row()
        append_row(
            replace(
                row, variant_name=name, provenance=replace(row.provenance, config_hash=config_hash)
            ),
            store,
        )
        prefix = (qid_prefix or {}).get(name, "q")
        write_predictions_rows(
            PredictionRows(
                confidence=base_confidence + shift,
                correct=predicted == gold,
                gold=gold,
                predicted=predicted,
                options=None,
                qid=np.asarray([f"{prefix}{item}" for item in range(N_ITEMS)], dtype=np.str_),
            ),
            root=root,
            key=predictions_key(config_hash, 0, "mmlu"),
        )


def _table(
    root: Path, plan: CalibrationSetPlan = PLAN
) -> tuple[list[CorpusShift], list[ShiftContrast], list[Skipped]]:
    tidy = load_tidy(ResultStore(root))
    return calibration_set_table(tidy, root=root, plan=plan, n_resamples=RESAMPLES)


def test_the_repo_plan_pairs_both_methods_and_names_the_contrast() -> None:
    plan = load_plan(REPO_PLAN)
    assert plan == PLAN


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("contrast: {first: a, second: b}\n", "no top-level 'pairs'"),
        ("pairs: {a-ood: a}\ncontrast: {first: a-ood}\n", "needs 'first' and 'second'"),
        ("pairs: {a-ood: a}\ncontrast: {first: a-ood, second: b-ood}\n", "no entry under"),
    ],
)
def test_a_malformed_plan_is_refused(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "plan.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_plan(path)


def test_shifts_and_contrast_recover_the_seeded_confidence_move(tmp_path: Path) -> None:
    _seed(tmp_path, CONFIDENCE_SHIFT)
    shifts, contrasts, skipped = _table(tmp_path)
    assert skipped == []
    by_variant = {shift.variant: shift for shift in shifts}
    assert by_variant["int4-awq-ood"].delta_confidence.point == pytest.approx(SHIFT)
    assert by_variant["int4-gptq-ood"].delta_confidence.point == pytest.approx(0.0)
    assert by_variant["int4-awq-ood"].delta_accuracy.point == pytest.approx(0.0)
    assert by_variant["int4-awq-ood"].answer_agreement == 1.0
    (contrast,) = contrasts
    assert (contrast.first, contrast.second) == ("int4-awq-ood", "int4-gptq-ood")
    assert contrast.confidence.point == pytest.approx(SHIFT)
    assert contrast.confidence.excludes_zero


def test_a_missing_twin_is_skipped_with_its_reason(tmp_path: Path) -> None:
    present = {name: shift for name, shift in CONFIDENCE_SHIFT.items() if name != "int4-awq"}
    _seed(tmp_path, present)
    shifts, contrasts, skipped = _table(tmp_path)
    assert [shift.variant for shift in shifts] == ["int4-gptq-ood"]
    assert contrasts == []
    assert any("int4-awq has no row" in skip.reason for skip in skipped)


def test_a_multi_seed_twin_is_skipped_rather_than_pooled(tmp_path: Path) -> None:
    _seed(tmp_path, CONFIDENCE_SHIFT)
    store = ResultStore(tmp_path)
    gptq_hash = "0" * 64
    row = sample_row()
    append_row(
        replace(
            row,
            variant_name="int4-gptq",
            provenance=replace(row.provenance, config_hash=gptq_hash, seed=1),
        ),
        store,
    )
    write_predictions_rows(
        PredictionRows(
            confidence=np.full(N_ITEMS, 0.7),
            correct=np.ones(N_ITEMS, dtype=bool),
            gold=np.zeros(N_ITEMS, dtype=np.intp),
            predicted=np.zeros(N_ITEMS, dtype=np.intp),
            options=None,
            qid=np.asarray([f"q{item}" for item in range(N_ITEMS)], dtype=np.str_),
        ),
        root=tmp_path,
        key=predictions_key(gptq_hash, 1, "mmlu"),
    )
    shifts, contrasts, skipped = _table(tmp_path)
    assert [shift.variant for shift in shifts] == ["int4-awq-ood"]
    assert contrasts == []
    assert any("int4-gptq has 2 seeds" in skip.reason for skip in skipped)


def test_twins_over_different_items_raise(tmp_path: Path) -> None:
    _seed(tmp_path, CONFIDENCE_SHIFT, qid_prefix={"int4-gptq-ood": "other"})
    with pytest.raises(ValueError, match="cannot pair int4-gptq-ood with int4-gptq"):
        _table(tmp_path)


def test_command_writes_the_shift_and_contrast_tables(tmp_path: Path) -> None:
    _seed(tmp_path, CONFIDENCE_SHIFT)
    result = CliRunner().invoke(
        app,
        [
            "calibration-set",
            "--results", str(tmp_path),
            "--plan", str(REPO_PLAN),
            "--resamples", str(RESAMPLES),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    shifts = pd.read_parquet(tmp_path / "calibration_set.parquet")
    contrasts = pd.read_parquet(tmp_path / "calibration_set_contrast.parquet")
    assert sorted(shifts["variant"]) == ["int4-awq-ood", "int4-gptq-ood"]
    assert contrasts["confidence_shift_difference"].iloc[0] == pytest.approx(SHIFT)
