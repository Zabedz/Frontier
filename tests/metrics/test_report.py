"""Report assembly: fields match the underlying calls and map onto Quality."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import make_calibrated_gold, make_softmax

from frontier.metrics.bootstrap import accuracy_ci, ece_ci
from frontier.metrics.calibration import (
    DEFAULT_SWEEP,
    ece_equal_mass,
    ece_equal_width,
    top_label,
)
from frontier.metrics.report import calibration_report, to_quality
from frontier.metrics.scoring import brier_decomposition, brier_score, nll
from frontier.schema import Quality


def test_report_fields_match_direct_calls() -> None:
    rng = np.random.default_rng(5)
    probs = make_softmax(2000, 4, rng, sharpness=1.4)
    gold = make_calibrated_gold(probs, rng)
    report = calibration_report(probs, gold, n_bins=10)
    _confidence, correct = top_label(probs, gold)
    decomposition = brier_decomposition(probs, gold, n_bins=10)
    assert report.accuracy == pytest.approx(float(np.mean(correct)))
    assert report.ece_equal_width == pytest.approx(ece_equal_width(probs, gold, n_bins=10))
    assert report.ece_equal_mass_ace == pytest.approx(ece_equal_mass(probs, gold, n_bins=10))
    assert tuple(report.ece_bin_sweep.keys()) == DEFAULT_SWEEP
    assert report.brier == pytest.approx(brier_score(probs, gold))
    assert report.brier_reliability == pytest.approx(decomposition.reliability)
    assert report.brier_resolution == pytest.approx(decomposition.resolution)
    assert report.brier_uncertainty == pytest.approx(decomposition.uncertainty)
    assert report.nll == pytest.approx(nll(probs, gold))
    assert int(report.reliability.count.sum()) == gold.shape[0]


def test_to_quality_populates_every_field_and_passes_through_inputs() -> None:
    rng = np.random.default_rng(6)
    probs = make_softmax(2000, 4, rng, sharpness=1.4)
    gold = make_calibrated_gold(probs, rng)
    report = calibration_report(probs, gold, n_bins=10)
    confidence, correct = top_label(probs, gold)
    accuracy_interval = accuracy_ci(correct, n_resamples=999, rng=1)
    ece_interval = ece_ci(confidence, correct, n_bins=10, n_resamples=999, rng=1)
    perplexity = 12.5
    temperature = 1.5

    quality = to_quality(
        report,
        accuracy_ci=accuracy_interval,
        ece_ci=ece_interval,
        perplexity=perplexity,
        temperature=temperature,
        temperature_scaled=True,
    )

    assert isinstance(quality, Quality)
    assert quality.accuracy == report.accuracy
    assert quality.accuracy_ci_low == accuracy_interval.low
    assert quality.accuracy_ci_high == accuracy_interval.high
    assert quality.ece_equal_width == report.ece_equal_width
    assert quality.ece_equal_mass_ace == report.ece_equal_mass_ace
    assert quality.ece_bin_sweep == report.ece_bin_sweep
    assert quality.ece_ci_low == ece_interval.low
    assert quality.ece_ci_high == ece_interval.high
    assert quality.brier == report.brier
    assert quality.brier_reliability == report.brier_reliability
    assert quality.brier_resolution == report.brier_resolution
    assert quality.brier_uncertainty == report.brier_uncertainty
    assert quality.nll == report.nll
    assert quality.perplexity == perplexity
    assert quality.temperature == temperature
    assert quality.temperature_scaled is True
    for field in dataclasses.fields(quality):
        assert getattr(quality, field.name) is not None
