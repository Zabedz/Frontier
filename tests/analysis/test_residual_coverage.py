"""The simulated pairs behind `frontier residual-coverage`, and the command itself."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from frontier.analysis import residual_coverage
from frontier.analysis.residual_coverage import (
    DEFAULT_SCENARIOS,
    TARGET_ACCURACY,
    TARGET_CONFIDENCE,
    CoverageRow,
    Geometry,
    Scenario,
    fit_geometry,
    run_scenario,
    simulate_pair,
    true_gap,
    variant_scale,
)
from frontier.metrics.bootstrap import ResidualInterval
from frontier.metrics.calibration import top_label
from frontier.pipeline.cli import app

LARGE = 200_000
TINY = 400
TRUTH_TINY = 20_000
NOISE_4BIT = Scenario("noise", 0.3)
REPLICATES = 2
ACCURACY_LOSS = -0.02  # the 4-bit rows lose 0.0235 to 0.0275
CONFIDENCE_HELD = 0.005
GAP_FLOOR = 0.003  # truth-run noise at LARGE items
STRONGEST_GAP = 0.02


@pytest.fixture(scope="module")
def geometry() -> Geometry:
    return fit_geometry(np.random.default_rng(0))


def _shift(scenario: Scenario, geometry: Geometry) -> tuple[float, float, float, float]:
    """Reference accuracy and confidence, then the variant's deltas on each."""
    rng = np.random.default_rng(1)
    pair = simulate_pair(scenario, geometry, variant_scale(scenario, geometry, rng), LARGE, rng)
    ref_conf, ref_correct = top_label(pair.reference_report.probs, pair.reference_report.gold)
    var_conf, var_correct = top_label(pair.variant_report.probs, pair.variant_report.gold)
    return (
        float(ref_correct.mean()),
        float(ref_conf.mean()),
        float(var_correct.mean() - ref_correct.mean()),
        float(var_conf.mean() - ref_conf.mean()),
    )


def _truth(scenario: Scenario, geometry: Geometry) -> float:
    rng = np.random.default_rng(2)
    return true_gap(scenario, geometry, variant_scale(scenario, geometry, rng), rng, n_items=LARGE)


def test_reference_reproduces_the_banked_fp16_row(geometry: Geometry) -> None:
    accuracy, confidence, _, _ = _shift(NOISE_4BIT, geometry)
    assert accuracy == pytest.approx(TARGET_ACCURACY, abs=0.01)
    assert confidence == pytest.approx(TARGET_CONFIDENCE, abs=0.005)


def test_noise_costs_accuracy_at_unchanged_confidence(geometry: Geometry) -> None:
    _, _, d_accuracy, d_confidence = _shift(NOISE_4BIT, geometry)
    assert d_accuracy < ACCURACY_LOSS
    assert abs(d_confidence) < CONFIDENCE_HELD


def test_noise_leaves_no_residual_gap(geometry: Geometry) -> None:
    assert abs(_truth(NOISE_4BIT, geometry)) < GAP_FLOOR


def test_spread_gap_grows_with_strength(geometry: Geometry) -> None:
    gaps = [_truth(Scenario("spread", strength), geometry) for strength in (0.1, 0.2, 0.3)]
    assert gaps == sorted(gaps)
    assert gaps[-1] > STRONGEST_GAP


def test_run_scenario_is_seeded_and_bounded(geometry: Geometry) -> None:
    def once() -> CoverageRow:
        return run_scenario(
            NOISE_4BIT,
            geometry,
            np.random.default_rng(3),
            replicates=REPLICATES,
            n_items=TINY,
            n_resamples=19,
            truth_items=TRUTH_TINY,
        )

    row = once()
    assert row == once()
    assert row.replicates == REPLICATES
    for share in (row.coverage, row.above_zero, row.below_zero):
        assert 0.0 <= share <= 1.0
    assert row.above_zero + row.below_zero <= 1.0
    assert row.mean_width >= 0.0


def _interval(point: float, low: float, high: float, *, refused: int = 0) -> ResidualInterval:
    return ResidualInterval(
        point=point, low=low, high=high, refused_resamples=refused, n_resamples=9
    )


def test_run_scenario_counts_against_the_truth(
    geometry: Geometry, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth = 0.005
    intervals: Iterator[ResidualInterval] = iter(
        [
            _interval(0.0, -0.01, 0.01),
            _interval(0.01, 0.002, 0.03, refused=1),
            _interval(-0.01, -0.03, -0.001),
            _interval(0.02, 0.006, 0.04),
        ]
    )

    def scripted(*_: Any, **__: Any) -> ResidualInterval:
        return next(intervals)

    def fixed_truth(*_: Any, **__: Any) -> float:
        return truth

    monkeypatch.setattr(residual_coverage, "paired_residual_ece_ci", scripted)
    monkeypatch.setattr(residual_coverage, "true_gap", fixed_truth)
    row = run_scenario(
        NOISE_4BIT, geometry, np.random.default_rng(4), replicates=4, n_items=TINY, n_resamples=9
    )
    assert row.true_gap == truth
    assert row.coverage == pytest.approx(2 / 4)
    assert row.above_zero == pytest.approx(2 / 4)
    assert row.below_zero == pytest.approx(1 / 4)
    assert row.mean_point == pytest.approx(0.02 / 4)
    assert row.mean_width == pytest.approx((0.02 + 0.028 + 0.029 + 0.034) / 4)
    assert row.unusable == 1


def test_command_writes_one_row_per_scenario(tmp_path: Path) -> None:
    out = tmp_path / "coverage.parquet"
    result = CliRunner().invoke(
        app,
        [
            "residual-coverage",
            "--replicates", "1",
            "--resamples", "9",
            "--items", str(TINY),
            "--truth-items", str(TRUTH_TINY),
            "--out", str(out),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    frame = pd.read_parquet(out)
    assert list(frame["scenario"]) == [scenario.name for scenario in DEFAULT_SCENARIOS]
    assert (frame["replicates"] == 1).all()
