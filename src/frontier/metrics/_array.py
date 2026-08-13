"""Shared array dtypes for the metric core, and the guards that reject a malformed batch
at the entry point, naming the first offending index and its value.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ProbMatrix = npt.NDArray[np.float64]  # (n_items, n_classes), rows sum to 1
FloatArray = npt.NDArray[np.float64]  # 1-D per-item or per-bin floats
LabelArray = npt.NDArray[np.intp]  # (n_items,), gold class ids in [0, n_classes)
CorrectArray = npt.NDArray[np.bool_]  # (n_items,), argmax == gold
IntArray = npt.NDArray[np.intp]  # per-bin counts, bin indices

_MATRIX_NDIM = 2


def check_predictions(probs: ProbMatrix, gold: LabelArray, *, atol: float = 1e-4) -> None:
    """Guard the raw prediction inputs shared by the ECE, Brier, and NLL paths.

    ``atol`` is the tolerance on the per-row simplex check.
    """
    if probs.ndim != _MATRIX_NDIM:
        raise ValueError(f"probs must be 2-D (n_items, n_classes), got shape {probs.shape}")
    if gold.ndim != 1:
        raise ValueError(f"gold must be 1-D (n_items,), got shape {gold.shape}")
    n_items, n_classes = probs.shape
    if gold.shape[0] != n_items:
        raise ValueError(f"probs has {n_items} rows but gold has {gold.shape[0]} labels")
    if n_items == 0:
        raise ValueError("empty input: probs has zero rows")
    off_simplex = np.abs(probs.sum(axis=1) - 1.0) > atol
    if bool(off_simplex.any()):
        row = int(np.argmax(off_simplex))
        raise ValueError(
            f"row {row} sums to {float(probs[row].sum())}, off 1.0 by more than atol={atol}"
        )
    out_of_range = (gold < 0) | (gold >= n_classes)
    if bool(out_of_range.any()):
        item = int(np.argmax(out_of_range))
        raise ValueError(f"gold[{item}] = {int(gold[item])} is outside [0, {n_classes})")


def check_confidence(confidence: FloatArray, correct: CorrectArray) -> None:
    """Guard the reduced 1-D confidence-space inputs of ``ece_from_confidence``."""
    if confidence.ndim != 1:
        raise ValueError(f"confidence must be 1-D, got shape {confidence.shape}")
    if correct.ndim != 1:
        raise ValueError(f"correct must be 1-D, got shape {correct.shape}")
    if confidence.shape[0] != correct.shape[0]:
        raise ValueError(
            f"confidence has {confidence.shape[0]} items but correct has {correct.shape[0]}"
        )
    if confidence.shape[0] == 0:
        raise ValueError("empty input: confidence has zero items")
    outside = (confidence < 0.0) | (confidence > 1.0)
    if bool(outside.any()):
        item = int(np.argmax(outside))
        raise ValueError(f"confidence[{item}] = {float(confidence[item])} is outside [0, 1]")
