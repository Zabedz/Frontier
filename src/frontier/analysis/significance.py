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
    """One variant measured against a reference on its own backend.

    ``crosses_model`` marks a family pair whose reference scores another model.
    """

    variant: str
    reference: str
    task: str
    backend: str
    track: str
    crosses_model: bool = False

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


@dataclass(frozen=True, slots=True)
class References:
    """The reference variant for each backend, with per-model overrides and family extras.

    ``default`` serves every model; ``by_model`` maps a model id to the backends whose
    reference differs for that model, as for the 0.5B student. ``also_by_family`` names a
    second reference for every variant of a family on a backend, which may score another
    model, as the teacher's fp16 row does for a distilled student.
    """

    default: Mapping[str, str]
    by_model: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    also_by_family: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def for_variant(self, backend: str, model_id: str) -> str | None:
        """The reference a variant on ``backend`` scoring ``model_id`` is measured against."""
        override = self.by_model.get(model_id, {})
        return override.get(backend, self.default.get(backend))

    def family_reference(self, family: str, backend: str) -> str | None:
        """The second reference for a variant of ``family`` on ``backend``, if configured."""
        return self.also_by_family.get(family, {}).get(backend)


def load_references(path: Path = DEFAULT_REFERENCES_PATH) -> References:
    """Read ``references`` (backend to variant) and the optional ``by_model`` overrides and
    ``also_by_family`` extras."""
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict) or "references" not in loaded:
        raise ValueError(f"{path} has no top-level 'references' mapping")
    nested = {name: loaded.get(name) or {} for name in ("by_model", "also_by_family")}
    for name, block in nested.items():
        if not isinstance(block, dict):
            raise ValueError(f"{path}: {name!r} must be a mapping, got {type(block).__name__}")
    return References(
        default=_backend_map(path, "references", loaded["references"]),
        by_model={
            str(model): _backend_map(path, f"by_model.{model}", mapping)
            for model, mapping in nested["by_model"].items()
        },
        also_by_family={
            str(family): _backend_map(path, f"also_by_family.{family}", mapping)
            for family, mapping in nested["also_by_family"].items()
        },
    )


def _backend_map(path: Path, where: str, mapping: object) -> dict[str, str]:
    if not isinstance(mapping, dict):
        raise ValueError(f"{path}: {where!r} must be a mapping, got {type(mapping).__name__}")
    for backend, variant in mapping.items():
        if not isinstance(variant, str):
            raise ValueError(
                f"{path}: {where} reference for backend {backend!r} must be a variant name, "
                f"got {variant!r}"
            )
    return {str(backend): str(variant) for backend, variant in mapping.items()}


def resolve_pairs(
    tidy: pd.DataFrame, references: References | Mapping[str, str]
) -> tuple[list[VariantPair], list[Skipped]]:
    """Pair every variant with the reference for its backend and model, on each task.

    A plain mapping is the per-backend default for every model. The per-model pair skips a
    reference that scores a different model from the variant. A family reference from
    ``also_by_family`` follows as a second pair, unless the first already names it, and may
    cross models: both score the same items. A reference is one deterministic run, so one
    scored at several seeds is skipped. A variant may carry any number of training seeds,
    each scoring the reference's items.
    """
    if not isinstance(references, References):
        references = References(default=references)
    pairs: list[VariantPair] = []
    skipped: list[Skipped] = []
    if tidy.empty:
        return pairs, skipped
    seeds = _seeds_by_variant(tidy)
    models = _model_by_variant(tidy)
    for _, subset in tidy.groupby(["variant_name", "task_name"], sort=False):
        first = subset.iloc[0]
        group = _Group(
            variant=str(first["variant_name"]),
            task=str(first["task_name"]),
            backend=str(first["backend"]),
            track=str(first["track"]),
            model=str(first["model_id"]),
        )
        primary = _primary_pair(
            group,
            references.for_variant(group.backend, group.model),
            seeds=seeds,
            models=models,
        )
        outcomes = [primary]
        extra = references.family_reference(str(first["family"]), group.backend)
        paired = isinstance(primary, VariantPair) and primary.reference == extra
        if extra is not None and extra != group.variant and not paired:
            outcomes.append(_family_pair(group, extra, seeds=seeds, models=models))
        for outcome in outcomes:
            if isinstance(outcome, VariantPair):
                pairs.append(outcome)
            else:
                skipped.append(outcome)
    return pairs, skipped


@dataclass(frozen=True, slots=True)
class _Group:
    """One (variant, task) group of the tidy frame."""

    variant: str
    task: str
    backend: str
    track: str
    model: str

    def pair(self, reference: str, *, crosses_model: bool = False) -> VariantPair:
        return VariantPair(
            variant=self.variant,
            reference=reference,
            task=self.task,
            backend=self.backend,
            track=self.track,
            crosses_model=crosses_model,
        )


def _primary_pair(
    group: _Group,
    reference: str | None,
    *,
    seeds: Mapping[tuple[str, str], frozenset[int]],
    models: Mapping[tuple[str, str], str],
) -> VariantPair | Skipped:
    variant, task = group.variant, group.task
    if reference is None:
        return Skipped(variant, task, f"no reference configured for backend {group.backend}")
    if variant == reference:
        return Skipped(variant, task, "is the reference for its backend")
    if (reference, task) not in seeds:
        return Skipped(variant, task, f"reference {reference} has no row on task {task}")
    if models[(reference, task)] != group.model:
        return Skipped(
            variant,
            task,
            f"reference {reference} scores {models[(reference, task)]}, not {group.model}",
        )
    if len(seeds[(reference, task)]) != 1:
        return Skipped(
            variant, task, _several_seeds("reference", reference, seeds[(reference, task)])
        )
    return group.pair(reference)


def _family_pair(
    group: _Group,
    reference: str,
    *,
    seeds: Mapping[tuple[str, str], frozenset[int]],
    models: Mapping[tuple[str, str], str],
) -> VariantPair | Skipped:
    variant, task = group.variant, group.task
    if (reference, task) not in seeds:
        return Skipped(variant, task, f"family reference {reference} has no row on task {task}")
    if len(seeds[(reference, task)]) != 1:
        return Skipped(
            variant, task, _several_seeds("family reference", reference, seeds[(reference, task)])
        )
    return group.pair(reference, crosses_model=models[(reference, task)] != group.model)


def _several_seeds(kind: str, reference: str, seeds: frozenset[int]) -> str:
    return (
        f"{kind} {reference} is scored at seeds {sorted(seeds)}; a reference is one "
        f"deterministic run"
    )


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
    references: References | Mapping[str, str],
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
            "crosses_model": item.pair.crosses_model,
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


def _model_by_variant(tidy: pd.DataFrame) -> dict[tuple[str, str], str]:
    return {
        (str(variant), str(task)): str(subset["model_id"].iloc[0])
        for (variant, task), subset in tidy.groupby(["variant_name", "task_name"], sort=False)
    }


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
