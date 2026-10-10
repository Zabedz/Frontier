"""How well confidence separates right answers from wrong ones, which ECE does not measure:
a model can be miscalibrated by a constant offset and still rank its answers well.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata  # type: ignore[import-untyped]

from frontier.metrics._array import CorrectArray, FloatArray, check_confidence


def auroc(confidence: FloatArray, correct: CorrectArray) -> float:
    """Area under the ROC curve of ``confidence`` as a predictor of ``correct``.

    The probability that a correct item carries more confidence than an incorrect one, ties
    counting one half (the Mann-Whitney form). ``nan`` when either class is empty.
    """
    check_confidence(confidence, correct)
    n_correct = int(np.count_nonzero(correct))
    n_wrong = int(correct.shape[0]) - n_correct
    if n_correct == 0 or n_wrong == 0:
        return math.nan
    ranks = rankdata(confidence)
    rank_sum = float(np.sum(ranks[correct]))
    return (rank_sum - n_correct * (n_correct + 1) / 2.0) / (n_correct * n_wrong)
