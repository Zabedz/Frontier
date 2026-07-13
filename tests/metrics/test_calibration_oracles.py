"""Oracle cross-checks: our ECE family against torchmetrics, netcal, and sklearn.

Each oracle is guarded by ``pytest.importorskip`` so the base environment stays
green without the ``oracles`` group. The oracle handles come back through
``importorskip`` (typed ``Any``) rather than a top-level import, so neither the
missing package nor its absent type stubs break collection or mypy.

The fixture uses gold correlated with the softmax (a genuine reliability curve), so
the matched-setting agreement is tight and the ECE is bin-sensitive enough for the
final sensitivity guard to mean something. Confidences are continuous and never
exactly 1.0, so our left-closed binning agrees with torchmetrics everywhere.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from conftest import make_calibrated_gold, make_softmax

from frontier.metrics.binning import bin_edges
from frontier.metrics.calibration import ece, ece_equal_mass, ece_equal_width, top_label

OracleData = tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]

N_BINS = 15
N_CLASSES = 4


@pytest.fixture
def oracle_data() -> OracleData:
    rng = np.random.default_rng(2026)
    probs = make_softmax(3000, N_CLASSES, rng, sharpness=1.4)
    gold = make_calibrated_gold(probs, rng)
    return probs, gold


def test_torchmetrics_matches_equal_width(oracle_data: OracleData) -> None:
    probs, gold = oracle_data
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torchmetrics.functional")
    oracle = float(
        functional.calibration_error(
            torch.tensor(probs),
            torch.tensor(gold),
            task="multiclass",
            num_classes=N_CLASSES,
            n_bins=N_BINS,
            norm="l1",
        )
    )
    assert oracle == pytest.approx(ece_equal_width(probs, gold, n_bins=N_BINS), abs=1e-6)


def test_netcal_ece_matches_equal_width(oracle_data: OracleData) -> None:
    probs, gold = oracle_data
    metrics = pytest.importorskip("netcal.metrics")
    oracle = float(metrics.ECE(bins=N_BINS).measure(probs, gold))
    assert oracle == pytest.approx(ece_equal_width(probs, gold, n_bins=N_BINS), abs=1e-6)


def test_netcal_adaptive_ece_matches_equal_mass(oracle_data: OracleData) -> None:
    probs, gold = oracle_data
    metrics = pytest.importorskip("netcal.metrics")
    oracle = float(metrics.ECE(bins=N_BINS, equal_intervals=False).measure(probs, gold))
    assert oracle == pytest.approx(ece_equal_mass(probs, gold, n_bins=N_BINS), abs=1e-6)


def test_netcal_ace_matches_equal_mass_equal_weight(oracle_data: OracleData) -> None:
    probs, gold = oracle_data
    metrics = pytest.importorskip("netcal.metrics")
    oracle = float(metrics.ACE(bins=N_BINS, equal_intervals=False).measure(probs, gold))
    ours = ece(probs, gold, scheme="equal_mass", weighting="equal", n_bins=N_BINS)
    assert oracle == pytest.approx(ours, abs=5e-3)


def test_netcal_ace_matches_equal_width_equal_weight(oracle_data: OracleData) -> None:
    probs, gold = oracle_data
    metrics = pytest.importorskip("netcal.metrics")
    oracle = float(metrics.ACE(bins=N_BINS).measure(probs, gold))
    ours = ece(probs, gold, scheme="equal_width", weighting="equal", n_bins=N_BINS)
    assert oracle == pytest.approx(ours, abs=1e-6)


def test_sklearn_uniform_reduction_matches_equal_width(oracle_data: OracleData) -> None:
    probs, gold = oracle_data
    calibration = pytest.importorskip("sklearn.calibration")
    confidence, correct = top_label(probs, gold)
    prob_true, prob_pred = calibration.calibration_curve(
        correct.astype(int), confidence, n_bins=N_BINS, strategy="uniform"
    )
    counts = np.histogram(confidence, bins=np.linspace(0.0, 1.0, N_BINS + 1))[0]
    populated = counts > 0
    mass = counts[populated] / confidence.shape[0]
    reduced = float(np.sum(mass * np.abs(prob_true - prob_pred)))
    assert reduced == pytest.approx(ece_equal_width(probs, gold, n_bins=N_BINS), abs=1e-9)


def test_sklearn_quantile_reduction_matches_equal_mass(oracle_data: OracleData) -> None:
    probs, gold = oracle_data
    calibration = pytest.importorskip("sklearn.calibration")
    confidence, correct = top_label(probs, gold)
    prob_true, prob_pred = calibration.calibration_curve(
        correct.astype(int), confidence, n_bins=N_BINS, strategy="quantile"
    )
    edges = bin_edges(confidence, N_BINS, "equal_mass")
    counts = np.histogram(confidence, bins=edges)[0]
    populated = counts > 0
    mass = counts[populated] / confidence.shape[0]
    reduced = float(np.sum(mass * np.abs(prob_true - prob_pred)))
    assert reduced == pytest.approx(ece_equal_mass(probs, gold, n_bins=N_BINS), abs=1e-9)


def test_ece_is_sensitive_to_bin_count(oracle_data: OracleData) -> None:
    probs, gold = oracle_data
    metrics = pytest.importorskip("netcal.metrics")
    ours_fine = ece_equal_width(probs, gold, n_bins=15)
    oracle_coarse = float(metrics.ECE(bins=10).measure(probs, gold))
    assert ours_fine != pytest.approx(oracle_coarse, abs=1e-6)
