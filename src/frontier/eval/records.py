"""The normalised multiple-choice record and the shared array dtypes.

The aliases match the dtypes ``frontier.metrics`` uses, so arrays hand across without a copy
or a cast; they are declared here to keep the two packages decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ProbMatrix = npt.NDArray[np.float64]  # (n_items, max_options), rows sum to 1, zero-padded
FloatArray = npt.NDArray[np.float64]  # 1-D per-item floats, or a vocab-length logit row
LabelArray = npt.NDArray[np.intp]  # (n_items,), option indices
IntArray = npt.NDArray[np.intp]  # per-item option counts, candidate token ids
CorrectArray = npt.NDArray[np.bool_]  # (n_items,), predicted == gold

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MIN_OPTIONS = 2
MAX_OPTIONS = len(LETTERS)


@dataclass(frozen=True, slots=True)
class EvalRecord:
    """One normalised multiple-choice item, dataset-agnostic.

    ``gold`` indexes ``options`` in the record's canonical order, which is never permuted.
    ``error_type`` and ``redux_gold`` are populated only by the MMLU-Redux loader, where
    ``redux_gold`` is ``None`` for an item the policy drops.
    """

    qid: str
    question: str
    options: tuple[str, ...]
    gold: int
    subject: str
    split: str
    error_type: str | None = None
    redux_gold: int | None = None


def letters_for(n_options: int) -> str:
    """The first ``n_options`` answer letters, e.g. 4 -> ``"ABCD"``; 2..26 only."""
    if n_options < MIN_OPTIONS:
        raise ValueError(
            f"n_options must be >= {MIN_OPTIONS} for a multiple-choice item, got {n_options}"
        )
    if n_options > MAX_OPTIONS:
        raise ValueError(
            f"n_options must be <= {MAX_OPTIONS} (one answer letter per option), got {n_options}"
        )
    return LETTERS[:n_options]
