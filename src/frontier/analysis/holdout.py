"""The held-out split that post-hoc recalibration is fitted and reported on.

Pre-committed in ``docs/methodology.md`` section 8, ahead of any result, because a split
chosen after seeing the numbers is not a held-out split.

Membership is keyed on the item's ``qid``, hashed. ``load_slice`` subsamples per seed, so
one item sits at different offsets in different seeds' sidecars; a positional rule would
put it in the fit half for one seed and the report half for another. Hashing the id makes
membership a property of the item, so it holds across seeds, subset sizes, and variants,
and is independent of pooling order.

The hash spreads across subjects too, which matters because ``loaders.subsample`` returns
dataset order and MMLU is grouped by subject.
"""

from __future__ import annotations

from hashlib import blake2b

import numpy as np
import numpy.typing as npt

from frontier.io.predictions import IntArray, OptionProbs, PredictionRows, QidArray

HOLDOUT_STRIDE = 10
FIT_POSITIONS = 3  # 30% fits the temperature, 70% reports it

BoolArray = npt.NDArray[np.bool_]


class NotRecalibratableError(ValueError):
    """A variant that cannot be recalibrated, for a reason that is not corruption."""


def is_fit(qid: str) -> bool:
    """Whether ``qid`` belongs to the fit half.

    ``blake2b`` is stable across processes. The builtin ``hash`` is salted per process and
    would move the split between runs.
    """
    digest = blake2b(qid.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % HOLDOUT_STRIDE < FIT_POSITIONS


def fit_mask(qids: QidArray) -> BoolArray:
    """Per-item membership of the fit half."""
    mask: BoolArray = np.asarray([is_fit(str(qid)) for qid in qids], dtype=np.bool_)
    return mask


def take(rows: PredictionRows, positions: IntArray) -> PredictionRows:
    """The sub-slice of one sidecar at ``positions``, carrying every column along."""
    options = (
        None
        if rows.options is None
        else OptionProbs(
            probs=rows.options.probs[positions], n_options=rows.options.n_options[positions]
        )
    )
    return PredictionRows(
        confidence=rows.confidence[positions],
        correct=rows.correct[positions],
        gold=rows.gold[positions],
        predicted=rows.predicted[positions],
        options=options,
        qid=None if rows.qid is None else rows.qid[positions],
    )


def split(rows: PredictionRows) -> tuple[PredictionRows, PredictionRows]:
    """Split into ``(fit, report)`` by item id.

    Raises ``NotRecalibratableError`` when the sidecar carries no ids, or when the hash leaves
    either half empty.
    """
    if rows.qid is None:
        raise NotRecalibratableError(
            "sidecar carries no qid, so the split cannot be keyed on the item; it predates "
            "the qid column and has to be re-run to be recalibrated"
        )
    mask = fit_mask(rows.qid)
    positions = np.arange(rows.qid.shape[0])
    fit, report = positions[mask], positions[~mask]
    if fit.size == 0 or report.size == 0:
        raise NotRecalibratableError(
            f"the split left a half empty ({fit.size} fit, {report.size} report) over "
            f"{positions.size} items"
        )
    return take(rows, fit), take(rows, report)
