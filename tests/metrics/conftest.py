"""Synthetic generators shared across the metric tests.

Every generator takes an explicit ``np.random.Generator``, so a failure reproduces.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
LabelArray = npt.NDArray[np.intp]

# Values sit more than a width-0.1 bin apart, so at n_bins >= 10 binned Brier equals raw Brier.
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
    """Two-class softmax whose ``top_label`` reduction recovers the calibrated sample.

    ``low`` stays at or above 0.5, or the second column becomes the argmax.
    """
    confidence, correct = make_calibrated_confidence(n, rng, low=low)
    probs: FloatArray = np.column_stack([confidence, 1.0 - confidence])
    gold = np.where(correct, 0, 1).astype(np.intp)
    return probs, gold, confidence, correct


def make_softmax(
    n: int, n_classes: int, rng: np.random.Generator, *, sharpness: float = 1.0
) -> FloatArray:
    """Random-logit softmax rows: valid, continuous, and never exactly 1.0.

    ``sharpness`` scales the logits, which moves the average confidence up.
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

    The reliability curve stays non-degenerate, which keeps the oracle ECE bin-sensitive.
    Uniform gold flattens accuracy across confidence and collapses the bin sweep.
    """
    n, n_classes = probs.shape
    argmax = probs.argmax(axis=1)
    hit = rng.uniform(0.0, 1.0, size=n) < probs.max(axis=1)
    other = (argmax + rng.integers(1, n_classes, size=n)) % n_classes
    return np.where(hit, argmax, other).astype(np.intp)


def make_discrete_softmax(n: int, rng: np.random.Generator) -> tuple[FloatArray, LabelArray]:
    """Softmax rows drawn from the bin-isolated prototype set, with random gold."""
    index = rng.integers(0, DISCRETE_PROTOTYPES.shape[0], size=n)
    probs: FloatArray = DISCRETE_PROTOTYPES[index]
    gold = rng.integers(0, DISCRETE_PROTOTYPES.shape[1], size=n).astype(np.intp)
    return probs, gold


def make_skewed_crossing(n: int, rng: np.random.Generator) -> tuple[FloatArray, LabelArray]:
    """Skewed-high confidence with a reliability curve that crosses the diagonal.

    The calibration gap changes sign inside a wide bin, which separates equal-width
    from equal-mass on both ECE and the count of populated bins.
    """
    confidence = np.clip(rng.beta(5.0, 1.5, size=n), 0.5, 1.0 - 1e-9)
    p_correct = np.clip(1.6 * confidence - 0.55, 0.0, 1.0)
    correct = rng.uniform(0.0, 1.0, size=n) < p_correct
    probs: FloatArray = np.column_stack([confidence, 1.0 - confidence])
    gold = np.where(correct, 0, 1).astype(np.intp)
    return probs, gold
