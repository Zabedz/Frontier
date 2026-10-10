"""Paired significance over banked rows: whether compression damages calibration beyond the
accuracy it costs.

A stored row keeps one single-variant interval, so the paired statistics run on the
per-item sidecars. A pairing over mismatched items is silent, so ``_check_alignment``
compares the stored gold vectors before any resampling.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from frontier.analysis._skipped import Skipped
from frontier.analysis.load import load_predictions_for_variant, load_seed_predictions
from frontier.io.predictions import PredictionRows
from frontier.metrics.bootstrap import (
    DEFAULT_RESAMPLES,
    ConfidenceInterval,
    RatioInterval,
    SeededQuantity,
    paired_damage_gap_ci,
    paired_damage_ratio_ci,
    paired_delta_accuracy_ci,
    paired_delta_confidence_ci,
    paired_delta_ece_ci,
    paired_seeded_ci,
    paired_seeded_hierarchical_ci,
    paired_seeded_ratio_ci,
    seeded_quantity,
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

    ``delta_*``: absolute differences, variant minus reference. ``excess_damage`` is the
    verdict statistic: the ECE rise beyond the accuracy lost (see ``seeded_quantity``),
    zero when ECE moves only because answers flipped. ``delta_auroc`` is the change in how
    well confidence separates right answers from wrong. ``damage_gap`` and ``damage_ratio``
    are relative and kept for reference only: ECE sits well below accuracy, so a pure
    accuracy loss already drives the gap above zero and the ratio to accuracy over ECE.
    ``delta_ece_sweep`` repeats the ECE delta at each sweep bin count plus the headline
    count, since an ECE delta that changes sign with the bin count supports no claim.

    At several seeds, ``across_seeds`` holds the intervals that also resample the seed
    copies; the verdict properties read those, since the question-only intervals are
    conditional on the copies in hand.
    """

    pair: VariantPair
    n_items: int
    n_bins: int
    delta_accuracy: ConfidenceInterval
    delta_confidence: ConfidenceInterval
    delta_ece: ConfidenceInterval
    excess_damage: ConfidenceInterval
    delta_auroc: ConfidenceInterval
    damage_gap: ConfidenceInterval
    damage_ratio: RatioInterval
    delta_ece_sweep: dict[int, ConfidenceInterval]
    n_seeds: int = 1
    # Per-seed point estimates as (min, max), one entry per quantity; empty at one seed.
    seed_spread: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Seed-and-question intervals, keyed by quantity; empty at one seed.
    across_seeds: dict[str, ConfidenceInterval] = field(default_factory=dict)

    def verdict_interval(self, quantity: SeededQuantity) -> ConfidenceInterval:
        """The interval a claim about the method rests on: across seeds when there are any."""
        if quantity in self.across_seeds:
            return self.across_seeds[quantity]
        interval: ConfidenceInterval = getattr(self, quantity)
        return interval

    @property
    def confidence_shifted(self) -> bool:
        """Whether compression moved what the model says about itself, beyond noise."""
        return self.verdict_interval("delta_confidence").excludes_zero

    @property
    def calibration_damaged(self) -> bool:
        """Whether ECE rose beyond the accuracy lost, beyond noise."""
        interval = self.verdict_interval("excess_damage")
        return interval.excludes_zero and interval.point > 0.0

    @property
    def discrimination_moved(self) -> bool:
        """Whether confidence ranks right against wrong answers differently, beyond noise."""
        return self.verdict_interval("delta_auroc").excludes_zero

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

    A reference is one deterministic run, so one scored at several seeds is skipped. A
    variant may carry any number of training seeds, each scoring the reference's items.
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
        if len(seeds[(reference, task)]) != 1:
            skipped.append(
                Skipped(
                    variant,
                    task,
                    f"reference {reference} is scored at seeds "
                    f"{sorted(seeds[(reference, task)])}; a reference is one deterministic run",
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
    delta_confidence = paired_delta_confidence_ci(
        reference_rows.confidence,
        variant_rows.confidence,
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
    reference = (reference_rows.confidence, reference_rows.correct)
    variant = [(variant_rows.confidence, variant_rows.correct)]
    excess_damage, delta_auroc = (
        paired_seeded_ci(
            reference,
            variant,
            quantity=quantity,
            n_bins=n_bins,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            rng=rng,
        )
        for quantity in ("excess_damage", "delta_auroc")
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
        delta_confidence=delta_confidence,
        delta_ece=sweep[n_bins],
        excess_damage=excess_damage,
        delta_auroc=delta_auroc,
        damage_gap=damage_gap,
        damage_ratio=damage_ratio,
        delta_ece_sweep=sweep,
    )


_SPREAD_QUANTITIES: tuple[SeededQuantity, ...] = (
    "delta_accuracy",
    "delta_confidence",
    "delta_ece",
    "excess_damage",
    "delta_auroc",
    "damage_gap",
)
# The quantities a verdict reads; each gets a seed-and-question interval at several seeds.
_VERDICT_QUANTITIES: tuple[SeededQuantity, ...] = (
    "delta_accuracy",
    "delta_confidence",
    "delta_ece",
    "excess_damage",
    "delta_auroc",
)


def pair_significance_seeded(
    pair: VariantPair,
    reference_rows: PredictionRows,
    seed_rows: Sequence[PredictionRows],
    *,
    n_bins: int = DEFAULT_BINS,
    sweep_bins: Sequence[int] = DEFAULT_SWEEP,
    confidence_level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: int | None = 0,
) -> PairSignificance:
    """``pair_significance`` with the variant averaged over its seeds.

    Every seed copy is checked against the reference's items. ``seed_spread`` carries each
    quantity's per-seed range and ``across_seeds`` the intervals that resample seeds too.
    """
    for rows in seed_rows:
        _check_alignment(pair, reference_rows, rows)
    reference = (reference_rows.confidence, reference_rows.correct)
    seeds = [(rows.confidence, rows.correct) for rows in seed_rows]

    def interval(quantity: SeededQuantity, bins: int = n_bins) -> ConfidenceInterval:
        return paired_seeded_ci(
            reference,
            seeds,
            quantity=quantity,
            n_bins=bins,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            rng=rng,
        )

    sweep = {bins: interval("delta_ece", bins) for bins in sorted({*sweep_bins, n_bins})}
    spread: dict[str, tuple[float, float]] = {}
    for quantity in _SPREAD_QUANTITIES:
        per_seed = [seeded_quantity(quantity, reference, [seed], n_bins=n_bins) for seed in seeds]
        spread[quantity] = (min(per_seed), max(per_seed))
    across: dict[str, ConfidenceInterval] = {
        quantity: paired_seeded_hierarchical_ci(
            reference,
            seeds,
            quantity=quantity,
            n_bins=n_bins,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            rng=rng,
        )
        for quantity in _VERDICT_QUANTITIES
    }
    return PairSignificance(
        pair=pair,
        n_items=int(reference_rows.gold.shape[0]),
        n_bins=n_bins,
        delta_accuracy=interval("delta_accuracy"),
        delta_confidence=interval("delta_confidence"),
        delta_ece=sweep[n_bins],
        excess_damage=interval("excess_damage"),
        delta_auroc=interval("delta_auroc"),
        damage_gap=interval("damage_gap"),
        damage_ratio=paired_seeded_ratio_ci(
            reference,
            seeds,
            n_bins=n_bins,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            rng=rng,
        ),
        delta_ece_sweep=sweep,
        n_seeds=len(seeds),
        seed_spread=spread,
        across_seeds=across,
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
    seeds = _seeds_by_variant(tidy)
    results: list[PairSignificance] = []
    for pair in pairs:
        n_seeds = len(seeds[(pair.variant, pair.task)])
        try:
            reference_rows = load_predictions_for_variant(
                tidy, variant_name=pair.reference, task_name=pair.task, root=root
            )
            if n_seeds == 1:
                variant_rows = load_predictions_for_variant(
                    tidy, variant_name=pair.variant, task_name=pair.task, root=root
                )
            else:
                seed_rows = load_seed_predictions(
                    tidy, variant_name=pair.variant, task_name=pair.task, root=root
                )
        except ValueError as missing:
            skipped.append(Skipped(pair.variant, pair.task, str(missing)))
            continue
        if n_seeds == 1:
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
        else:
            results.append(
                pair_significance_seeded(
                    pair,
                    reference_rows,
                    seed_rows,
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
            "delta_confidence": item.delta_confidence.point,
            "delta_confidence_low": item.delta_confidence.low,
            "delta_confidence_high": item.delta_confidence.high,
            "confidence_shifted": item.confidence_shifted,
            "delta_ece": item.delta_ece.point,
            "delta_ece_low": item.delta_ece.low,
            "delta_ece_high": item.delta_ece.high,
            "excess_damage": item.excess_damage.point,
            "excess_damage_low": item.excess_damage.low,
            "excess_damage_high": item.excess_damage.high,
            "calibration_damaged": item.calibration_damaged,
            "delta_auroc": item.delta_auroc.point,
            "delta_auroc_low": item.delta_auroc.low,
            "delta_auroc_high": item.delta_auroc.high,
            "discrimination_moved": item.discrimination_moved,
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
            "n_seeds": item.n_seeds,
            "seed_spread": json.dumps(
                {name: list(bounds) for name, bounds in item.seed_spread.items()}
            ),
            "across_seeds": json.dumps(
                {
                    name: [interval.low, interval.point, interval.high]
                    for name, interval in item.across_seeds.items()
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
