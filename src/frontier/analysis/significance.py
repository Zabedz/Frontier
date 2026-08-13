"""Paired significance over banked rows: whether calibration degrades faster than accuracy.

A stored row keeps one single-variant interval, so the paired statistics run on the
per-item sidecars. A pairing over mismatched items is silent, so ``_check_alignment``
compares the stored gold vectors before any resampling.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from frontier.analysis._skipped import Skipped
from frontier.analysis.load import load_predictions_for_variant
from frontier.io.predictions import PredictionRows
from frontier.metrics.bootstrap import (
    DEFAULT_RESAMPLES,
    ConfidenceInterval,
    RatioInterval,
    paired_damage_gap_ci,
    paired_damage_ratio_ci,
    paired_delta_accuracy_ci,
    paired_delta_ece_ci,
)
from frontier.metrics.calibration import DEFAULT_BINS, DEFAULT_SWEEP

DEFAULT_REFERENCES_PATH = Path("configs/analysis/significance.yaml")


@dataclass(frozen=True, slots=True)
class VariantPair:
    """One variant measured against the reference on its own backend."""

    variant: str
    reference: str
    task: str
    backend: str
    track: str

    @property
    def label(self) -> str:
        return f"{self.variant} vs {self.reference}"


@dataclass(frozen=True, slots=True)
class PairSignificance:
    """Every paired statistic for one variant against its reference.

    ``delta_*``: absolute differences, variant minus reference. ``damage_gap`` and
    ``damage_ratio`` put the two losses on one relative scale, the gap answering whether
    calibration degrades faster and the ratio by what multiple. ``delta_ece_sweep``
    repeats the ECE delta at each sweep bin count plus the headline count, since an ECE
    delta that changes sign with the bin count supports no claim at a single count.
    """

    pair: VariantPair
    n_items: int
    n_bins: int
    delta_accuracy: ConfidenceInterval
    delta_ece: ConfidenceInterval
    damage_gap: ConfidenceInterval
    damage_ratio: RatioInterval
    delta_ece_sweep: dict[int, ConfidenceInterval]

    @property
    def delta_ece_sign_stable(self) -> bool:
        """Whether the ECE delta keeps one sign across every bin count in the sweep."""
        points = [interval.point for interval in self.delta_ece_sweep.values()]
        if not points:
            return False
        return all(point > 0.0 for point in points) or all(point < 0.0 for point in points)

    @property
    def delta_ece_sweep_all_exclude_zero(self) -> bool:
        """Whether the ECE delta interval excludes zero at every bin count."""
        intervals = list(self.delta_ece_sweep.values())
        return bool(intervals) and all(interval.excludes_zero for interval in intervals)


def load_references(path: Path = DEFAULT_REFERENCES_PATH) -> dict[str, str]:
    """Read the backend-to-reference-variant map."""
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict) or "references" not in loaded:
        raise ValueError(f"{path} has no top-level 'references' mapping")
    references = loaded["references"]
    if not isinstance(references, dict):
        raise ValueError(f"{path}: 'references' must be a mapping, got {type(references).__name__}")
    for backend, variant in references.items():
        if not isinstance(variant, str):
            raise ValueError(
                f"{path}: reference for backend {backend!r} must be a variant name, got {variant!r}"
            )
    return {str(backend): str(variant) for backend, variant in references.items()}


def resolve_pairs(
    tidy: pd.DataFrame, references: Mapping[str, str]
) -> tuple[list[VariantPair], list[Skipped]]:
    """Pair every variant with the reference for its backend, on each task separately.

    A variant carrying a different seed set from its reference is skipped: the sidecar
    pooling concatenates seeds, so unequal sets would misalign the pairing.
    """
    pairs: list[VariantPair] = []
    skipped: list[Skipped] = []
    if tidy.empty:
        return pairs, skipped
    seeds = _seeds_by_variant(tidy)
    for _, subset in tidy.groupby(["variant_name", "task_name"], sort=False):
        first = subset.iloc[0]
        variant = str(first["variant_name"])
        task = str(first["task_name"])
        backend = str(first["backend"])
        reference = references.get(backend)
        if reference is None:
            skipped.append(Skipped(variant, task, f"no reference configured for backend {backend}"))
            continue
        if variant == reference:
            skipped.append(Skipped(variant, task, "is the reference for its backend"))
            continue
        if (reference, task) not in seeds:
            skipped.append(
                Skipped(variant, task, f"reference {reference} has no row on task {task}")
            )
            continue
        if seeds[(reference, task)] != seeds[(variant, task)]:
            skipped.append(
                Skipped(
                    variant,
                    task,
                    f"seed sets differ: {variant} has {sorted(seeds[(variant, task)])}, "
                    f"{reference} has {sorted(seeds[(reference, task)])}",
                )
            )
            continue
        pairs.append(
            VariantPair(
                variant=variant,
                reference=reference,
                task=task,
                backend=backend,
                track=str(first["track"]),
            )
        )
    return pairs, skipped


def pair_significance(
    pair: VariantPair,
    reference_rows: PredictionRows,
    variant_rows: PredictionRows,
    *,
    n_bins: int = DEFAULT_BINS,
    sweep_bins: Sequence[int] = DEFAULT_SWEEP,
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: int | None = 0,
) -> PairSignificance:
    """Run every paired statistic for one pair, reference first, variant second.

    Raises ``ValueError`` unless the two sidecars describe the same items in the same
    order.
    """
    _check_alignment(pair, reference_rows, variant_rows)
    arrays = (
        reference_rows.confidence,
        reference_rows.correct,
        variant_rows.confidence,
        variant_rows.correct,
    )
    delta_accuracy = paired_delta_accuracy_ci(
        reference_rows.correct,
        variant_rows.correct,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        rng=rng,
    )
    damage_gap = paired_damage_gap_ci(
        *arrays,
        n_bins=n_bins,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        rng=rng,
    )
    damage_ratio = paired_damage_ratio_ci(
        *arrays,
        n_bins=n_bins,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        rng=rng,
    )
    # The headline count is read out of the sweep, so both share one set of resamples.
    sweep = {
        bins: paired_delta_ece_ci(
            *arrays,
            n_bins=bins,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            rng=rng,
        )
        for bins in sorted({*sweep_bins, n_bins})
    }
    return PairSignificance(
        pair=pair,
        n_items=int(reference_rows.gold.shape[0]),
        n_bins=n_bins,
        delta_accuracy=delta_accuracy,
        delta_ece=sweep[n_bins],
        damage_gap=damage_gap,
        damage_ratio=damage_ratio,
        delta_ece_sweep=sweep,
    )


def significance_table(
    tidy: pd.DataFrame,
    *,
    root: Path,
    references: Mapping[str, str],
    n_bins: int = DEFAULT_BINS,
    sweep_bins: Sequence[int] = DEFAULT_SWEEP,
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: int | None = 0,
) -> tuple[list[PairSignificance], list[Skipped]]:
    """Resolve the pairs from the frame and run every statistic on each.

    A missing sidecar, or a variant pooling rows scored under more than one config hash,
    is skipped with that reason. Sidecars that disagree about the items raise, since that
    is a corrupted store.
    """
    pairs, skipped = resolve_pairs(tidy, references)
    results: list[PairSignificance] = []
    for pair in pairs:
        try:
            reference_rows = load_predictions_for_variant(
                tidy, variant_name=pair.reference, task_name=pair.task, root=root
            )
            variant_rows = load_predictions_for_variant(
                tidy, variant_name=pair.variant, task_name=pair.task, root=root
            )
        except ValueError as missing:
            skipped.append(Skipped(pair.variant, pair.task, str(missing)))
            continue
        results.append(
            pair_significance(
                pair,
                reference_rows,
                variant_rows,
                n_bins=n_bins,
                sweep_bins=sweep_bins,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                rng=rng,
            )
        )
    return results, skipped


def to_frame(results: Sequence[PairSignificance]) -> pd.DataFrame:
    """One row per pair; the bin sweep rides in a JSON column, as the store's lists do."""
    records = [
        {
            "variant": item.pair.variant,
            "reference": item.pair.reference,
            "task": item.pair.task,
            "backend": item.pair.backend,
            "track": item.pair.track,
            "n_items": item.n_items,
            "n_bins": item.n_bins,
            "delta_accuracy": item.delta_accuracy.point,
            "delta_accuracy_low": item.delta_accuracy.low,
            "delta_accuracy_high": item.delta_accuracy.high,
            "delta_ece": item.delta_ece.point,
            "delta_ece_low": item.delta_ece.low,
            "delta_ece_high": item.delta_ece.high,
            "damage_gap": item.damage_gap.point,
            "damage_gap_low": item.damage_gap.low,
            "damage_gap_high": item.damage_gap.high,
            "damage_ratio": item.damage_ratio.point,
            "damage_ratio_low": item.damage_ratio.low,
            "damage_ratio_high": item.damage_ratio.high,
            "damage_ratio_usable": item.damage_ratio.usable,
            "damage_ratio_nonfinite_resamples": item.damage_ratio.nonfinite_resamples,
            "accuracy_damage": item.damage_ratio.denominator.point,
            "accuracy_damage_low": item.damage_ratio.denominator.low,
            "accuracy_damage_high": item.damage_ratio.denominator.high,
            "delta_ece_sign_stable": item.delta_ece_sign_stable,
            "delta_ece_sweep": json.dumps(
                {
                    str(bins): [interval.low, interval.point, interval.high]
                    for bins, interval in item.delta_ece_sweep.items()
                }
            ),
        }
        for item in results
    ]
    return pd.DataFrame.from_records(records)


def _seeds_by_variant(tidy: pd.DataFrame) -> dict[tuple[str, str], frozenset[int]]:
    seeds: dict[tuple[str, str], frozenset[int]] = {}
    for _, subset in tidy.groupby(["variant_name", "task_name"], sort=False):
        key = (str(subset["variant_name"].iloc[0]), str(subset["task_name"].iloc[0]))
        seeds[key] = frozenset(int(seed) for seed in subset["seed"])
    return seeds


def _check_alignment(
    pair: VariantPair, reference_rows: PredictionRows, variant_rows: PredictionRows
) -> None:
    reference_gold = reference_rows.gold
    variant_gold = variant_rows.gold
    if reference_gold.shape != variant_gold.shape:
        raise ValueError(
            f"cannot pair {pair.variant} with {pair.reference} on {pair.task}: sidecars hold "
            f"{variant_gold.shape[0]} and {reference_gold.shape[0]} items"
        )
    if not np.array_equal(reference_gold, variant_gold):
        first = int(np.flatnonzero(reference_gold != variant_gold)[0])
        raise ValueError(
            f"cannot pair {pair.variant} with {pair.reference} on {pair.task}: the sidecars "
            f"describe different items, first disagreeing at index {first} "
            f"({pair.reference} gold {reference_gold[first]}, {pair.variant} gold "
            f"{variant_gold[first]})"
        )
