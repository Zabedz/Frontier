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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path

import numpy as np
import pandas as pd

from frontier.analysis._skipped import Skipped
from frontier.analysis.holdout import NotRecalibratableError
from frontier.analysis.load import MixedConfigHashError, load_split_predictions
from frontier.analysis.significance import VariantPair, resolve_pairs
from frontier.io.predictions import MissingSidecarError, PredictionRows, QidArray
from frontier.metrics.bootstrap import (
    DEFAULT_RESAMPLES,
    ResidualInterval,
    ScoredItems,
    paired_residual_ece_ci,
    residual_ece,
)
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
        except (
            MissingSidecarError,
            MixedConfigHashError,
            NotRecalibratableError,
            TemperatureFitError,
        ) as reason:
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


@dataclass(frozen=True, slots=True)
class RepairabilityPair:
    """One variant's post-repair residual against its backend reference.

    The per-variant table says how much error survived; this says whether the difference
    between two variants' survivals is bigger than the noise. Methodology section 6 asks
    for that before either is called a finding.
    """

    pair: VariantPair
    n_bins: int
    n_report: int
    residual_variant: float
    residual_reference: float
    residual_gap: ResidualInterval

    @property
    def variant_is_less_repairable(self) -> bool:
        """Whether the variant's surviving error exceeds the reference's, beyond noise.

        False whenever a resample refused a fit. Those bounds are quantiles over the
        surviving draws, so they come back finite however many were dropped, and
        ``excludes_zero`` alone would read a thinned distribution as a verdict.
        """
        return (
            self.residual_gap.usable
            and self.residual_gap.excludes_zero
            and self.residual_gap.point > 0.0
        )


def _scored(rows: PredictionRows, label: str) -> ScoredItems:
    if rows.options is None:
        raise NotRecalibratableError(f"{label} has no stored distributions to recalibrate")
    return ScoredItems(probs=rows.options.probs, gold=rows.gold, n_options=rows.options.n_options)


def check_alignment(
    pair: VariantPair, reference: PredictionRows, variant: PredictionRows, *, half: str
) -> None:
    """Refuse a pair whose ``half`` holds different items on the two sides.

    The split is keyed on the item, so matched halves follow from matched item sets. Two
    variants scored at different subset sizes hold different sets, and their residuals are
    then measured on different questions. Both halves are checked: the interval shares one
    index vector across the two sides of the fit half as well as the report half.

    Raises ``NotRecalibratableError``; ids are compared when both sides carry them, and
    lengths otherwise.
    """
    if reference.qid is not None and variant.qid is not None:
        if not np.array_equal(reference.qid, variant.qid):
            raise NotRecalibratableError(
                f"{pair.variant} and {pair.reference} {half} halves hold different items "
                f"({fingerprint(variant.qid)} against {fingerprint(reference.qid)})"
            )
        return
    if reference.gold.shape[0] != variant.gold.shape[0]:
        raise NotRecalibratableError(
            f"{pair.variant} and {pair.reference} {half} halves differ in length "
            f"({variant.gold.shape[0]} against {reference.gold.shape[0]})"
        )


def repairability_pairs(
    tidy: pd.DataFrame,
    *,
    root: Path,
    references: Mapping[str, str],
    n_bins: int = DEFAULT_BINS,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: int | None = 0,
) -> tuple[list[RepairabilityPair], list[Skipped]]:
    """Every configured pair's residual difference, with its interval."""
    pairs, skipped = resolve_pairs(tidy, references)
    found: list[RepairabilityPair] = []
    for pair in pairs:
        try:
            reference_fit, reference_report = load_split_predictions(
                tidy, variant_name=pair.reference, task_name=pair.task, root=root
            )
            variant_fit, variant_report = load_split_predictions(
                tidy, variant_name=pair.variant, task_name=pair.task, root=root
            )
            check_alignment(pair, reference_report, variant_report, half="report")
            check_alignment(pair, reference_fit, variant_fit, half="fit")
            reference_fit_items = _scored(reference_fit, pair.reference)
            reference_report_items = _scored(reference_report, pair.reference)
            variant_fit_items = _scored(variant_fit, pair.variant)
            variant_report_items = _scored(variant_report, pair.variant)
            gap = paired_residual_ece_ci(
                reference_fit_items,
                reference_report_items,
                variant_fit_items,
                variant_report_items,
                n_bins=n_bins,
                n_resamples=n_resamples,
                rng=rng,
            )
            residual_variant = residual_ece(variant_fit_items, variant_report_items, n_bins=n_bins)
            residual_reference = residual_ece(
                reference_fit_items, reference_report_items, n_bins=n_bins
            )
        except (
            MissingSidecarError,
            MixedConfigHashError,
            NotRecalibratableError,
            TemperatureFitError,
        ) as reason:
            skipped.append(Skipped(pair.variant, pair.task, str(reason)))
            continue
        found.append(
            RepairabilityPair(
                pair=pair,
                n_bins=n_bins,
                n_report=int(variant_report.gold.shape[0]),
                residual_variant=residual_variant,
                residual_reference=residual_reference,
                residual_gap=gap,
            )
        )
    return found, skipped


def pairs_to_frame(found: Sequence[RepairabilityPair]) -> pd.DataFrame:
    """One row per pair."""
    return pd.DataFrame.from_records(
        [
            {
                "variant": item.pair.variant,
                "reference": item.pair.reference,
                "task": item.pair.task,
                "backend": item.pair.backend,
                "track": item.pair.track,
                "n_bins": item.n_bins,
                "n_report": item.n_report,
                "residual_variant": item.residual_variant,
                "residual_reference": item.residual_reference,
                "residual_gap": item.residual_gap.point,
                "residual_gap_low": item.residual_gap.low,
                "residual_gap_high": item.residual_gap.high,
                "residual_gap_excludes_zero": item.residual_gap.excludes_zero,
                "residual_gap_usable": item.residual_gap.usable,
                "refused_resamples": item.residual_gap.refused_resamples,
                "variant_is_less_repairable": item.variant_is_less_repairable,
            }
            for item in found
        ]
    )
