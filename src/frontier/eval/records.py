"""The normalised multiple-choice record and the shared array dtypes.

The npt aliases name the shapes the eval core passes to ``frontier.metrics``; they
are the same dtypes WP1 uses, so the arrays hand across without a copy or a cast.
They are declared here rather than reached out of the private ``frontier.metrics``
array module, so the two packages stay decoupled.
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

    ``gold`` indexes ``options`` in the record's canonical order (never permuted).
    ``error_type`` and ``redux_gold`` are populated only by the MMLU-Redux loader;
    ``redux_gold`` is the de-noised label, or ``None`` when the policy drops the item
    from the de-noised set. Every other loader leaves them ``None``.
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
    """The first ``n_options`` answer letters, e.g. 4 -> ``"ABCD"``.

    Raises ``ValueError`` if ``n_options`` is below 2 (not a choice) or above 26 (no
    single letter left), naming the offending count.
    """
    if n_options < MIN_OPTIONS:
        raise ValueError(
            f"n_options must be >= {MIN_OPTIONS} for a multiple-choice item, got {n_options}"
        )
    if n_options > MAX_OPTIONS:
        raise ValueError(
            f"n_options must be <= {MAX_OPTIONS} (one answer letter per option), got {n_options}"
        )
    return LETTERS[:n_options]
