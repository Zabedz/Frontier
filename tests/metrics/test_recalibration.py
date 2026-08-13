"""Temperature scaling: the logit equivalence, the padding mask, and the fit."""

from __future__ import annotations

import numpy as np
import pytest

from frontier.metrics._array import FloatArray, IntArray, ProbMatrix
from frontier.metrics.recalibration import (
    apply_temperature,
    fit_temperature,
    temperature_nll,
)

N_ITEMS = 800
N_CLASSES = 4


def _softmax(scores: FloatArray) -> ProbMatrix:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    normalised: ProbMatrix = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    return normalised


def _logits(n: int = N_ITEMS, k: int = N_CLASSES, *, seed: int = 0) -> FloatArray:
    drawn: FloatArray = np.random.default_rng(seed).normal(0.0, 2.0, size=(n, k))
    return drawn


def _full(n: int, k: int) -> IntArray:
    counts: IntArray = np.full(n, k, dtype=np.intp)
    return counts


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.7, 4.0])
def test_scaling_stored_probabilities_equals_scaling_the_original_logits(
    temperature: float,
) -> None:
    """The reason the sidecar stores probabilities and not logits.

    ``log(p) = z - logZ`` and the constant is the same across a row, so it cancels in the
    softmax. If this ever fails, the stored distribution is no longer a sufficient input
    for recalibration and the sidecar has to carry logits.
    """
    logits = _logits()
    probs = _softmax(logits)
    from_probs = apply_temperature(probs, temperature, _full(N_ITEMS, N_CLASSES))
    from_logits = _softmax(logits / temperature)
    assert np.allclose(from_probs, from_logits, atol=1e-12)


def test_temperature_of_one_returns_the_input_distribution() -> None:
    probs = _softmax(_logits())
    scaled = apply_temperature(probs, 1.0, _full(N_ITEMS, N_CLASSES))
    assert np.allclose(scaled, probs, atol=1e-12)


def test_rows_stay_on_the_simplex_at_every_temperature() -> None:
    probs = _softmax(_logits())
    for temperature in (0.1, 1.0, 9.0):
        scaled = apply_temperature(probs, temperature, _full(N_ITEMS, N_CLASSES))
        assert np.allclose(scaled.sum(axis=1), 1.0, atol=1e-12)


def test_a_high_temperature_softens_and_a_low_one_sharpens() -> None:
    probs = _softmax(_logits())
    counts = _full(N_ITEMS, N_CLASSES)
    base = probs.max(axis=1).mean()
    assert apply_temperature(probs, 5.0, counts).max(axis=1).mean() < base
    assert apply_temperature(probs, 0.2, counts).max(axis=1).mean() > base


def test_padding_columns_never_take_mass() -> None:
    """A padded cell is 0.0, whose log is -inf; it must stay at zero after scaling."""
    probs = np.zeros((3, 4))
    probs[0, :4] = _softmax(_logits(1, 4, seed=1))[0]
    probs[1, :3] = _softmax(_logits(1, 3, seed=2))[0]
    probs[2, :2] = _softmax(_logits(1, 2, seed=3))[0]
    counts = np.asarray([4, 3, 2], dtype=np.intp)
    scaled = apply_temperature(probs, 2.5, counts)
    assert scaled[1, 3] == 0.0
    assert scaled[2, 2] == 0.0 and scaled[2, 3] == 0.0
    assert np.allclose(scaled.sum(axis=1), 1.0, atol=1e-12)


def test_padding_does_not_change_the_live_columns() -> None:
    """A 3-option item scaled in a width-4 matrix matches the same item scaled alone."""
    row = _softmax(_logits(1, 3, seed=7))[0]
    padded = np.zeros((1, 4))
    padded[0, :3] = row
    wide = apply_temperature(padded, 3.0, np.asarray([3], dtype=np.intp))
    narrow = apply_temperature(row[None, :], 3.0, np.asarray([3], dtype=np.intp))
    assert np.allclose(wide[0, :3], narrow[0], atol=1e-12)


def test_fit_recovers_the_temperature_that_distorted_a_calibrated_sample() -> None:
    """Sharpening logits by a factor of ``s`` takes a temperature of ``s`` to undo.

    Dividing the logits by 0.5 doubles them, so the fit should land near 2.0.
    """
    rng = np.random.default_rng(11)
    logits = rng.normal(0.0, 2.0, size=(6000, N_CLASSES))
    probs = _softmax(logits)
    gold = np.asarray([rng.choice(N_CLASSES, p=row) for row in probs], dtype=np.intp)
    sharpened = _softmax(logits / 0.5)
    fitted = fit_temperature(sharpened, gold, _full(6000, N_CLASSES))
    assert fitted == pytest.approx(2.0, abs=0.1)


def test_the_fitted_temperature_beats_every_other_on_the_fitted_sample() -> None:
    rng = np.random.default_rng(12)
    logits = rng.normal(0.0, 2.0, size=(3000, N_CLASSES))
    probs = _softmax(logits)
    gold = np.asarray([rng.choice(N_CLASSES, p=row) for row in probs], dtype=np.intp)
    overconfident = _softmax(logits / 0.4)
    counts = _full(3000, N_CLASSES)
    fitted = fit_temperature(overconfident, gold, counts)
    best = temperature_nll(overconfident, gold, counts, fitted)
    for other in (0.2, 0.7, 1.0, 3.0):
        assert best <= temperature_nll(overconfident, gold, counts, other) + 1e-9


def test_a_calibrated_sample_fits_near_one() -> None:
    rng = np.random.default_rng(13)
    probs = _softmax(rng.normal(0.0, 2.0, size=(6000, N_CLASSES)))
    gold = np.asarray([rng.choice(N_CLASSES, p=row) for row in probs], dtype=np.intp)
    assert fit_temperature(probs, gold, _full(6000, N_CLASSES)) == pytest.approx(1.0, abs=0.1)


def test_a_non_positive_temperature_is_rejected() -> None:
    probs = _softmax(_logits(4, N_CLASSES))
    counts = _full(4, N_CLASSES)
    with pytest.raises(ValueError, match="must be positive"):
        apply_temperature(probs, 0.0, counts)
    with pytest.raises(ValueError, match="must be positive"):
        temperature_nll(probs, np.zeros(4, dtype=np.intp), counts, -1.0)


def test_fitting_an_empty_sample_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty input"):
        fit_temperature(
            np.zeros((0, N_CLASSES)),
            np.zeros(0, dtype=np.intp),
            np.zeros(0, dtype=np.intp),
        )


def test_a_fit_running_to_a_bound_is_rejected() -> None:
    """Returning the bound would read as convergence on a sample that cannot be scaled."""
    rng = np.random.default_rng(21)
    logits = rng.normal(0.0, 2.0, size=(4000, N_CLASSES))
    probs = _softmax(logits)
    gold = np.asarray([rng.choice(N_CLASSES, p=row) for row in probs], dtype=np.intp)
    far_outside = _softmax(logits / 0.02)  # optimum near T = 50, well past the upper bound
    with pytest.raises(ValueError, match="ran to the bound"):
        fit_temperature(far_outside, gold, _full(4000, N_CLASSES))


def test_one_item_with_no_mass_on_gold_is_rejected() -> None:
    """An infinite NLL at every temperature otherwise swings the fit by a factor of ten."""
    rng = np.random.default_rng(22)
    logits = rng.normal(0.0, 2.0, size=(600, N_CLASSES))
    probs = _softmax(logits)
    gold = np.asarray([rng.choice(N_CLASSES, p=row) for row in probs], dtype=np.intp)
    probs[0] = np.asarray([1.0, 0.0, 0.0, 0.0])
    gold[0] = 1
    with pytest.raises(ValueError, match=r"did not converge|not finite|outside"):
        fit_temperature(probs, gold, _full(600, N_CLASSES))


def test_a_gold_array_shorter_than_the_matrix_is_rejected() -> None:
    """Silently fitting on a prefix would report a temperature for the wrong sample."""
    probs = _softmax(_logits(400, N_CLASSES))
    with pytest.raises(ValueError):
        fit_temperature(probs, np.zeros(20, dtype=np.intp), _full(400, N_CLASSES))


def test_an_n_options_of_the_wrong_length_is_rejected() -> None:
    probs = _softmax(_logits(40, N_CLASSES))
    with pytest.raises(ValueError, match="n_options has"):
        apply_temperature(probs, 1.5, np.asarray([N_CLASSES], dtype=np.intp))


def test_an_option_count_below_one_is_rejected() -> None:
    probs = _softmax(_logits(3, N_CLASSES))
    with pytest.raises(ValueError, match="at least one option"):
        apply_temperature(probs, 1.5, np.asarray([4, 0, 4], dtype=np.intp))


def test_junk_beyond_the_option_count_is_masked_out() -> None:
    """Padding is only inert because of the mask; zeros alone would hide a broken mask."""
    probs = np.zeros((2, 4))
    probs[0, :4] = _softmax(_logits(1, 4, seed=5))[0]
    probs[1, :3] = _softmax(_logits(1, 3, seed=6))[0]
    probs[1, 3] = 0.4  # junk past n_options, which a zero-only fixture would never catch
    counts = np.asarray([4, 3], dtype=np.intp)
    scaled = apply_temperature(probs, 2.0, counts)
    assert scaled[1, 3] == 0.0
    assert np.allclose(scaled.sum(axis=1), 1.0, atol=1e-12)


def test_the_fit_agrees_with_netcal_temperature_scaling() -> None:
    """Oracle check, the rule every other metric in this package follows.

    netcal parameterises the scale as a multiplier on the logits and stores the reciprocal
    of the temperature in this module's divisor convention. The transform comparison is the
    one that does not depend on whose convention is whose: netcal's own output at its
    fitted scale must equal ``apply_temperature`` at ours.
    """
    netcal_scaling = pytest.importorskip("netcal.scaling")
    rng = np.random.default_rng(31)
    logits = rng.normal(0.0, 2.0, size=(4000, N_CLASSES))
    probs = _softmax(logits)
    gold = np.asarray([rng.choice(N_CLASSES, p=row) for row in probs], dtype=np.intp)
    overconfident = _softmax(logits / 0.45)
    counts = _full(4000, N_CLASSES)

    ours = fit_temperature(overconfident, gold, counts)
    oracle = netcal_scaling.TemperatureScaling()
    oracle.fit(overconfident.astype(np.float64), gold.astype(np.int64))
    theirs = float(np.ravel(oracle.temperature)[0])

    assert ours == pytest.approx(1.0 / theirs, rel=1e-3)
    assert np.allclose(
        oracle.transform(overconfident.astype(np.float64)),
        apply_temperature(overconfident, ours, counts),
        atol=1e-4,
    )
