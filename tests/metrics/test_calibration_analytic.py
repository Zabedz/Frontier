"""Known-answer calibration fixtures and the input guards."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    FloatArray,
    LabelArray,
    make_calibrated_softmax,
    make_gold,
    make_skewed_crossing,
    make_softmax,
)

from frontier.metrics.binning import bin_stats
from frontier.metrics.calibration import (
    DEFAULT_SWEEP,
    ece,
    ece_equal_mass,
    ece_equal_width,
    ece_from_confidence,
    ece_sweep,
    reliability_curve,
    top_label,
)


def _onehot(gold: LabelArray, n_classes: int, chosen: LabelArray) -> FloatArray:
    probs = np.zeros((gold.shape[0], n_classes), dtype=np.float64)
    probs[np.arange(gold.shape[0]), chosen] = 1.0
    return probs


def test_all_confident_correct_has_zero_ece_and_full_accuracy() -> None:
    rng = np.random.default_rng(0)
    gold = make_gold(500, 4, rng)
    probs = _onehot(gold, 4, gold)
    assert ece_equal_width(probs, gold) == 0.0
    assert ece_equal_mass(probs, gold) == 0.0
    assert float(np.mean(top_label(probs, gold)[1])) == 1.0


def test_all_confident_wrong_has_unit_ece_and_zero_accuracy() -> None:
    rng = np.random.default_rng(1)
    gold = make_gold(500, 4, rng)
    probs = _onehot(gold, 4, (gold + 1) % 4)
    assert ece_equal_width(probs, gold) == 1.0
    assert float(np.mean(top_label(probs, gold)[1])) == 0.0


def test_two_bin_hand_computation() -> None:
    confidence = np.array([0.2, 0.4, 0.6, 0.8])
    correct = np.array([False, True, True, True])
    equal_width = ece_from_confidence(confidence, correct, n_bins=2, scheme="equal_width")
    equal_mass = ece_from_confidence(confidence, correct, n_bins=2, scheme="equal_mass")
    assert equal_width == pytest.approx(0.25)
    assert equal_mass == pytest.approx(0.25)


def test_equal_weighting_coincides_with_mass_under_equal_bins() -> None:
    confidence = np.array([0.2, 0.4, 0.6, 0.8])
    correct = np.array([False, True, True, True])
    mass = ece_from_confidence(confidence, correct, n_bins=2, weighting="mass")
    equal = ece_from_confidence(confidence, correct, n_bins=2, weighting="equal")
    assert mass == pytest.approx(equal)


def test_large_calibrated_sample_has_small_positive_bin_bias() -> None:
    rng = np.random.default_rng(20260713)
    probs, gold, _confidence, _correct = make_calibrated_softmax(20000, rng)
    observed = ece_equal_width(probs, gold)  # observed 0.005394 at this seed
    bias_ceiling = 0.02
    assert 0.0 < observed < bias_ceiling


def test_bin_sweep_keys_finite_and_bias_does_not_shrink_with_finer_bins() -> None:
    rng = np.random.default_rng(20260713)
    probs, gold, _confidence, _correct = make_calibrated_softmax(20000, rng)
    sweep = ece_sweep(probs, gold)
    assert tuple(sweep.keys()) == DEFAULT_SWEEP
    for value in sweep.values():
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0
    assert sweep[30] >= sweep[10] - 1e-3


def test_equal_mass_spreads_bins_and_shifts_ece_on_a_crossing_curve() -> None:
    probs, gold = make_skewed_crossing(6000, np.random.default_rng(101))
    confidence, correct = top_label(probs, gold)
    populated_width = int((bin_stats(confidence, correct, 15, "equal_width").count > 0).sum())
    populated_mass = int((bin_stats(confidence, correct, 15, "equal_mass").count > 0).sum())
    assert populated_mass > populated_width
    equal_width = ece_equal_width(probs, gold, n_bins=15)
    equal_mass = ece_equal_mass(probs, gold, n_bins=15)
    min_gap = 1e-3
    assert abs(equal_width - equal_mass) > min_gap


def test_reliability_curve_marks_empty_bins_with_nan() -> None:
    rng = np.random.default_rng(9)
    probs = make_softmax(1000, 5, rng)
    gold = make_gold(1000, 5, rng)
    curve = reliability_curve(probs, gold, n_bins=10)
    empty = curve.count == 0
    populated = curve.count > 0
    assert bool(empty.any())
    assert bool(np.all(np.isnan(curve.mean_confidence[empty])))
    assert bool(np.all(np.isnan(curve.accuracy[empty])))
    assert bool(np.all(~np.isnan(curve.accuracy[populated])))
    assert int(curve.count.sum()) == gold.shape[0]


def test_check_predictions_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty input"):
        ece_equal_width(np.zeros((0, 4)), np.array([], dtype=np.intp))


def test_check_predictions_rejects_ragged_shapes() -> None:
    probs = make_softmax(10, 4, np.random.default_rng(0))
    with pytest.raises(ValueError, match="rows but gold"):
        ece_equal_width(probs, make_gold(9, 4, np.random.default_rng(0)))


def test_check_predictions_rejects_off_simplex_rows() -> None:
    probs = make_softmax(10, 4, np.random.default_rng(0))
    probs[3] *= 2.0
    with pytest.raises(ValueError, match="row 3 sums"):
        ece_equal_width(probs, make_gold(10, 4, np.random.default_rng(0)))


def test_check_predictions_rejects_out_of_range_gold() -> None:
    probs = make_softmax(10, 4, np.random.default_rng(0))
    gold = make_gold(10, 4, np.random.default_rng(0))
    gold[2] = 4
    with pytest.raises(ValueError, match=r"gold\[2\]"):
        ece_equal_width(probs, gold)


def test_check_predictions_rejects_non_matrix_probs() -> None:
    with pytest.raises(ValueError, match="probs must be 2-D"):
        ece(np.array([0.5, 0.5]), np.array([0], dtype=np.intp))


def test_check_predictions_rejects_non_vector_gold() -> None:
    probs = make_softmax(10, 4, np.random.default_rng(0))
    with pytest.raises(ValueError, match="gold must be 1-D"):
        ece(probs, np.zeros((10, 1), dtype=np.intp))


def test_check_confidence_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="confidence has"):
        ece_from_confidence(np.array([0.5, 0.6]), np.array([True]))


def test_check_confidence_rejects_non_vector_confidence() -> None:
    with pytest.raises(ValueError, match="confidence must be 1-D"):
        ece_from_confidence(np.zeros((2, 2)), np.array([True, False]))


def test_check_confidence_rejects_non_vector_correct() -> None:
    with pytest.raises(ValueError, match="correct must be 1-D"):
        ece_from_confidence(np.array([0.5, 0.6]), np.array([[True], [False]]))


def test_check_confidence_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty input"):
        ece_from_confidence(np.array([]), np.array([], dtype=np.bool_))


def test_check_confidence_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match=r"confidence\[1\]"):
        ece_from_confidence(np.array([0.5, 1.5]), np.array([True, False]))
