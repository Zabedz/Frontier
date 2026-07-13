"""Seeded synthetic generators shared across the metric test suite.

The generators are deterministic: tests pass an explicit ``np.random.Generator``
so a failure reproduces. They build the calibration inputs the suite reasons about,
a genuinely calibrated top-label sample (``P(correct | conf=c) = c``), continuous
softmax rows that never reach 1.0, and a discrete-forecast fixture whose per-class
bin means equal its raw forecasts.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
LabelArray = npt.NDArray[np.intp]

# Discrete-forecast prototypes. Each distinct per-class value is separated from the
# others by more than the width-0.1 bin, so with n_bins >= 10 every distinct
# forecast is alone in its bin and the binned Brier reconstructs the raw Brier.
DISCRETE_PROTOTYPES: FloatArray = np.array(
    [
        [0.65, 0.25, 0.10],
        [0.25, 0.65, 0.10],
        [0.10, 0.25, 0.65],
        [0.45, 0.45, 0.10],
    ]
)


def make_calibrated_confidence(
    n: int, rng: np.random.Generator, *, low: float = 0.5
) -> tuple[FloatArray, BoolArray]:
    """Draw a calibrated top-label sample: ``conf ~ U(low, 1)``, ``correct ~ Bern(conf)``."""
    confidence: FloatArray = rng.uniform(low, 1.0, size=n)
    correct: BoolArray = rng.uniform(0.0, 1.0, size=n) < confidence
    return confidence, correct


def make_calibrated_softmax(
    n: int, rng: np.random.Generator, *, low: float = 0.5
) -> tuple[FloatArray, LabelArray, FloatArray, BoolArray]:
    """A two-class softmax whose top-label reduction is a calibrated sample.

    The max column is ``confidence`` (kept ``>= 0.5`` so it is the argmax) and the
    gold label is 0 exactly when the item is correct, so ``top_label`` recovers the
    same ``(confidence, correct)`` pair.
    """
    confidence, correct = make_calibrated_confidence(n, rng, low=low)
    probs: FloatArray = np.column_stack([confidence, 1.0 - confidence])
    gold = np.where(correct, 0, 1).astype(np.intp)
    return probs, gold, confidence, correct


def make_softmax(
    n: int, n_classes: int, rng: np.random.Generator, *, sharpness: float = 1.0
) -> FloatArray:
    """Random-logit softmax rows: valid, continuous, and never exactly 1.0.

    ``sharpness`` scales the logits, moving the average confidence up for the
    skewed and oracle fixtures.
    """
    logits = rng.standard_normal((n, n_classes)) * sharpness
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs: FloatArray = exp / exp.sum(axis=1, keepdims=True)
    return probs


def make_gold(n: int, n_classes: int, rng: np.random.Generator) -> LabelArray:
    """Uniform random gold ids in ``[0, n_classes)``."""
    return rng.integers(0, n_classes, size=n).astype(np.intp)


def make_calibrated_gold(probs: FloatArray, rng: np.random.Generator) -> LabelArray:
    """Gold correlated with the softmax so ``P(correct | conf=c) ~ c``.

    Each item lands on its argmax class with probability equal to the max softmax,
    otherwise on a uniformly chosen other class. This gives a non-degenerate
    reliability curve, so the oracle ECE is genuinely bin-sensitive (unlike random
    gold, whose accuracy is flat across confidence and collapses the bin sweep).
    """
    n, n_classes = probs.shape
    argmax = probs.argmax(axis=1)
    hit = rng.uniform(0.0, 1.0, size=n) < probs.max(axis=1)
    other = (argmax + rng.integers(1, n_classes, size=n)) % n_classes
    return np.where(hit, argmax, other).astype(np.intp)


def make_discrete_softmax(n: int, rng: np.random.Generator) -> tuple[FloatArray, LabelArray]:
    """Softmax rows drawn from a small prototype set, with random gold.

    Every forecast value is bin-isolated, so the binned Brier equals the raw Brier.
    """
    index = rng.integers(0, DISCRETE_PROTOTYPES.shape[0], size=n)
    probs: FloatArray = DISCRETE_PROTOTYPES[index]
    gold = rng.integers(0, DISCRETE_PROTOTYPES.shape[1], size=n).astype(np.intp)
    return probs, gold


def make_skewed_crossing(n: int, rng: np.random.Generator) -> tuple[FloatArray, LabelArray]:
    """Skewed-high confidence with a reliability curve that crosses the diagonal.

    The calibration gap changes sign within the range, so equal-width bins that
    straddle the crossing cancel it while equal-mass bins resolve it: equal-mass
    then reports a different ECE and populates strictly more bins.
    """
    confidence = np.clip(rng.beta(5.0, 1.5, size=n), 0.5, 1.0 - 1e-9)
    p_correct = np.clip(1.6 * confidence - 0.55, 0.0, 1.0)
    correct = rng.uniform(0.0, 1.0, size=n) < p_correct
    probs: FloatArray = np.column_stack([confidence, 1.0 - confidence])
    gold = np.where(correct, 0, 1).astype(np.intp)
    return probs, gold
