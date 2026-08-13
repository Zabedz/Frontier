"""Per-variant repairability: how much of each variant's calibration error a single
temperature removes, and how much survives it.

The residual is the contribution. If one temperature closes the gap for every compression
family, the deployment story is "refit the threshold after compression"; a family whose
residual stays high is a family that argument does not cover.

The temperature is fitted on the held-out split's fit half and the numbers come from its
report half, so what is reported is the error that survived the repair.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path

import pandas as pd

from frontier.analysis._skipped import Skipped
from frontier.analysis.holdout import NotRecalibratableError
from frontier.analysis.load import load_split_predictions
from frontier.io.predictions import MissingSidecarError, PredictionRows, QidArray
from frontier.metrics.calibration import DEFAULT_BINS, ece_from_confidence, top_label
from frontier.metrics.recalibration import (
    TemperatureFitError,
    apply_temperature,
    fit_temperature,
    temperature_nll,
)
from frontier.metrics.scoring import brier_decomposition

REPORT_FLOOR = 1000  # methodology section 6


@dataclass(frozen=True, slots=True)
class Repairability:
    """One variant before and after a single fitted temperature, on the report half."""

    variant: str
    task: str
    backend: str
    family: str
    temperature: float
    n_bins: int
    n_fit: int
    n_report: int
    report_fingerprint: str  # of the report half's sorted qids; "" when the sidecar has none
    ece_before: float
    ece_after: float
    brier_reliability_before: float
    brier_reliability_after: float
    nll_before: float
    nll_after: float
    accuracy: float

    @property
    def ece_removed_fraction(self) -> float:
        """Share of the ECE the temperature removed, negative when it made things worse.

        ``nan`` on a zero denominator. A 10-bin ECE carries a finite-sample floor of its
        own (methodology section 3), so this ratio is unstable exactly where the residual
        is most interesting; read ``ece_before`` and ``ece_after`` for the real answer.
        """
        if self.ece_before == 0.0:
            return math.nan
        return (self.ece_before - self.ece_after) / self.ece_before

    @property
    def report_below_floor(self) -> bool:
        """Whether the report half falls under the >=1000-item floor in methodology 6."""
        return self.n_report < REPORT_FLOOR


def repairability(
    rows_fit: PredictionRows,
    rows_report: PredictionRows,
    *,
    variant: str,
    task: str,
    backend: str,
    family: str,
    n_bins: int = DEFAULT_BINS,
) -> Repairability:
    """Fit on ``rows_fit``, measure the residual on ``rows_report``.

    Accuracy is unchanged by temperature scaling, which is monotone within a row, so it is
    reported once as the report half's own accuracy.
    """
    if rows_fit.options is None or rows_report.options is None:
        raise NotRecalibratableError(
            f"variant {variant!r} has no stored distributions to recalibrate"
        )
    temperature = fit_temperature(rows_fit.options.probs, rows_fit.gold, rows_fit.options.n_options)
    before = rows_report.options.probs
    after = apply_temperature(before, temperature, rows_report.options.n_options)
    confidence_before, correct = top_label(before, rows_report.gold)
    confidence_after, _ = top_label(after, rows_report.gold)
    return Repairability(
        variant=variant,
        task=task,
        backend=backend,
        family=family,
        temperature=temperature,
        n_bins=n_bins,
        n_fit=int(rows_fit.gold.shape[0]),
        n_report=int(rows_report.gold.shape[0]),
        report_fingerprint=fingerprint(rows_report.qid),
        ece_before=ece_from_confidence(confidence_before, correct, n_bins=n_bins),
        ece_after=ece_from_confidence(confidence_after, correct, n_bins=n_bins),
        brier_reliability_before=brier_decomposition(
            before, rows_report.gold, n_bins=n_bins
        ).reliability,
        brier_reliability_after=brier_decomposition(
            after, rows_report.gold, n_bins=n_bins
        ).reliability,
        nll_before=temperature_nll(before, rows_report.gold, rows_report.options.n_options, 1.0),
        nll_after=temperature_nll(
            before, rows_report.gold, rows_report.options.n_options, temperature
        ),
        accuracy=float(correct.mean()),
    )


def repairability_table(
    tidy: pd.DataFrame, *, root: Path, n_bins: int = DEFAULT_BINS
) -> tuple[list[Repairability], list[Skipped]]:
    """Every variant in the frame that carries distributions, fitted and reported."""
    found: list[Repairability] = []
    skipped: list[Skipped] = []
    if tidy.empty:
        return found, skipped
    for _, subset in tidy.groupby(["variant_name", "task_name"], sort=False):
        first = subset.iloc[0]
        variant = str(first["variant_name"])
        task = str(first["task_name"])
        try:
            rows_fit, rows_report = load_split_predictions(
                tidy, variant_name=variant, task_name=task, root=root
            )
            found.append(
                repairability(
                    rows_fit,
                    rows_report,
                    variant=variant,
                    task=task,
                    backend=str(first["backend"]),
                    family=str(first["family"]),
                    n_bins=n_bins,
                )
            )
        except (MissingSidecarError, NotRecalibratableError, TemperatureFitError) as reason:
            skipped.append(Skipped(variant, task, str(reason)))
    return found, skipped


def fingerprint(qids: QidArray | None) -> str:
    """A stable digest of an item set, so two rows can be told apart or matched up.

    Two variants scored on different seeds or subset sizes hold different report halves,
    and nothing else in the table would say so.
    """
    if qids is None:
        return ""
    digest = blake2b(digest_size=8)
    for qid in sorted(str(item) for item in qids):
        digest.update(qid.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def to_frame(found: Sequence[Repairability]) -> pd.DataFrame:
    """One row per variant."""
    return pd.DataFrame.from_records(
        [
            {
                "variant": item.variant,
                "task": item.task,
                "backend": item.backend,
                "family": item.family,
                "temperature": item.temperature,
                "n_bins": item.n_bins,
                "n_fit": item.n_fit,
                "n_report": item.n_report,
                "report_fingerprint": item.report_fingerprint,
                "report_below_floor": item.report_below_floor,
                "accuracy": item.accuracy,
                "ece_before": item.ece_before,
                "ece_after": item.ece_after,
                "ece_removed_fraction": item.ece_removed_fraction,
                "brier_reliability_before": item.brier_reliability_before,
                "brier_reliability_after": item.brier_reliability_after,
                "nll_before": item.nll_before,
                "nll_after": item.nll_after,
            }
            for item in found
        ]
    )
