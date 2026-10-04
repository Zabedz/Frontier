"""Coverage and power of the paired residual-ECE interval on simulated pairs.

The reference reproduces the banked fp16 row (4 options, accuracy 0.647, mean confidence
0.915), and one temperature repairs it exactly. Each scenario derives a variant from the
same items and gold, so the pair stays paired, and fixes the true gap with one large run.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq  # type: ignore[import-untyped]

from frontier.analysis.holdout import FIT_POSITIONS, HOLDOUT_STRIDE
from frontier.metrics._array import FloatArray, IntArray, LabelArray, ProbMatrix
from frontier.metrics.bootstrap import ScoredItems, paired_residual_ece_ci, residual_ece

N_OPTIONS = 4
TARGET_ACCURACY = 0.647  # banked fp16, raw MMLU
TARGET_CONFIDENCE = 0.915  # banked fp16 mean top-label confidence
TRUTH_ITEMS = 1_000_000
CALIBRATION_ITEMS = 200_000
SCALE_BRACKET = (0.05, 50.0)

ScenarioKind = Literal["noise", "spread"]


@dataclass(frozen=True, slots=True)
class Scenario:
    """How the variant departs from the reference.

    ``noise`` adds Gaussian noise of ``strength`` latent standard deviations and rescales to
    hold mean confidence, so accuracy falls while confidence stays put, the pattern the
    banked pairs show; its true gap is near zero. ``spread`` scales a random half of the
    items by ``exp(strength)`` and the rest by ``exp(-strength)``, a mixed overconfidence no
    single temperature removes.

    A pure rescaling is not offered: its fitted temperatures differ by exactly the scale in
    every resample, so the interval collapses onto zero and coverage means nothing.
    """

    kind: ScenarioKind
    strength: float

    @property
    def name(self) -> str:
        return f"{self.kind}-{self.strength:g}"


DEFAULT_SCENARIOS = (
    Scenario("noise", 0.15),
    Scenario("noise", 0.3),
    Scenario("spread", 0.1),
    Scenario("spread", 0.15),
    Scenario("spread", 0.2),
    Scenario("spread", 0.3),
)


@dataclass(frozen=True, slots=True)
class Geometry:
    """Logit multipliers fitted to the targets.

    ``sharpness`` makes the calibrated model as accurate as fp16; ``overconfidence`` is the
    factor on top of it that a fitted temperature has to undo.
    """

    sharpness: float
    overconfidence: float


@dataclass(frozen=True, slots=True)
class SimulatedPair:
    reference_fit: ScoredItems
    reference_report: ScoredItems
    variant_fit: ScoredItems
    variant_report: ScoredItems


@dataclass(frozen=True, slots=True)
class CoverageRow:
    """One scenario's replicates against its true gap.

    ``above_zero`` and ``below_zero`` are the shares of intervals wholly on that side, so
    under a zero gap their sum is the false-positive rate and under a positive gap
    ``above_zero`` is the power. ``unusable`` counts replicates with a refused resample.
    """

    scenario: str
    true_gap: float
    replicates: int
    coverage: float
    above_zero: float
    below_zero: float
    mean_point: float
    mean_width: float
    unusable: int


def _softmax(logits: FloatArray) -> ProbMatrix:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    probs: ProbMatrix = weights / weights.sum(axis=1, keepdims=True)
    return probs


def _solve_scale(logits: FloatArray, target: float) -> float:
    """The multiplier on ``logits`` at which mean top-label confidence equals ``target``."""

    def excess(scale: float) -> float:
        return float(_softmax(scale * logits).max(axis=1).mean()) - target

    low, high = SCALE_BRACKET
    return float(brentq(excess, low, high))


def fit_geometry(rng: np.random.Generator) -> Geometry:
    """Solve both multipliers on one large latent sample.

    A calibrated model's accuracy equals its mean confidence, so ``sharpness`` is solved
    against the accuracy target.
    """
    latent = rng.standard_normal((CALIBRATION_ITEMS, N_OPTIONS))
    sharpness = _solve_scale(latent, TARGET_ACCURACY)
    overconfidence = _solve_scale(sharpness * latent, TARGET_CONFIDENCE)
    return Geometry(sharpness=sharpness, overconfidence=overconfidence)


def variant_scale(scenario: Scenario, geometry: Geometry, rng: np.random.Generator) -> float:
    """The multiplier on the variant's logits, fixed once per scenario."""
    if scenario.kind == "spread":
        return geometry.overconfidence
    latent = rng.standard_normal((CALIBRATION_ITEMS, N_OPTIONS))
    noise = rng.standard_normal(latent.shape)
    return _solve_scale(
        geometry.sharpness * (latent + scenario.strength * noise), TARGET_CONFIDENCE
    )


def _draw_gold(probs: ProbMatrix, rng: np.random.Generator) -> LabelArray:
    draws = rng.random(probs.shape[0])
    crossed = (draws[:, None] > probs.cumsum(axis=1)).sum(axis=1)
    gold: LabelArray = np.minimum(crossed, N_OPTIONS - 1).astype(np.intp)
    return gold


def _variant_logits(
    scenario: Scenario,
    latent: FloatArray,
    geometry: Geometry,
    scale: float,
    rng: np.random.Generator,
) -> FloatArray:
    if scenario.kind == "spread":
        signs = rng.choice(np.array([-1.0, 1.0]), size=latent.shape[0])
        per_item = scale * np.exp(scenario.strength * signs)
        spread: FloatArray = per_item[:, None] * geometry.sharpness * latent
        return spread
    noise = rng.standard_normal(latent.shape)
    noisy: FloatArray = scale * geometry.sharpness * (latent + scenario.strength * noise)
    return noisy


def _halves(probs: ProbMatrix, gold: LabelArray, n_fit: int) -> tuple[ScoredItems, ScoredItems]:
    n_options: IntArray = np.full(gold.shape[0], N_OPTIONS, dtype=np.intp)
    fit = ScoredItems(probs[:n_fit], gold[:n_fit], n_options[:n_fit])
    report = ScoredItems(probs[n_fit:], gold[n_fit:], n_options[n_fit:])
    return fit, report


def simulate_pair(
    scenario: Scenario,
    geometry: Geometry,
    scale: float,
    n_items: int,
    rng: np.random.Generator,
) -> SimulatedPair:
    """Draw ``n_items`` items, their gold from the calibrated model, and both variants.

    Items are independent, so the leading ``FIT_POSITIONS / HOLDOUT_STRIDE`` share is the
    fit half, the same proportion the qid hash gives the real sidecars.
    """
    latent = rng.standard_normal((n_items, N_OPTIONS))
    gold = _draw_gold(_softmax(geometry.sharpness * latent), rng)
    reference = _softmax(geometry.overconfidence * geometry.sharpness * latent)
    variant = _softmax(_variant_logits(scenario, latent, geometry, scale, rng))
    n_fit = n_items * FIT_POSITIONS // HOLDOUT_STRIDE
    reference_fit, reference_report = _halves(reference, gold, n_fit)
    variant_fit, variant_report = _halves(variant, gold, n_fit)
    return SimulatedPair(reference_fit, reference_report, variant_fit, variant_report)


def true_gap(
    scenario: Scenario,
    geometry: Geometry,
    scale: float,
    rng: np.random.Generator,
    *,
    n_items: int = TRUTH_ITEMS,
) -> float:
    """Residual of the variant minus residual of the reference on one large draw."""
    pair = simulate_pair(scenario, geometry, scale, n_items, rng)
    return residual_ece(pair.variant_fit, pair.variant_report) - residual_ece(
        pair.reference_fit, pair.reference_report
    )


def run_scenario(
    scenario: Scenario,
    geometry: Geometry,
    rng: np.random.Generator,
    *,
    replicates: int,
    n_items: int,
    n_resamples: int,
    truth_items: int = TRUTH_ITEMS,
) -> CoverageRow:
    """Score ``replicates`` intervals at the real sample size against the scenario's truth."""
    scale = variant_scale(scenario, geometry, rng)
    truth = true_gap(scenario, geometry, scale, rng, n_items=truth_items)
    covered = above = below = unusable = 0
    points: list[float] = []
    widths: list[float] = []
    for _ in range(replicates):
        pair = simulate_pair(scenario, geometry, scale, n_items, rng)
        interval = paired_residual_ece_ci(
            pair.reference_fit,
            pair.reference_report,
            pair.variant_fit,
            pair.variant_report,
            n_resamples=n_resamples,
            rng=rng,
        )
        covered += interval.low <= truth <= interval.high
        above += interval.low > 0.0
        below += interval.high < 0.0
        unusable += not interval.usable
        points.append(interval.point)
        widths.append(interval.high - interval.low)
    return CoverageRow(
        scenario=scenario.name,
        true_gap=truth,
        replicates=replicates,
        coverage=covered / replicates,
        above_zero=above / replicates,
        below_zero=below / replicates,
        mean_point=math.fsum(points) / replicates,
        mean_width=math.fsum(widths) / replicates,
        unusable=unusable,
    )


def study(
    *,
    replicates: int,
    n_items: int,
    n_resamples: int,
    seed: int,
    truth_items: int = TRUTH_ITEMS,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
) -> Iterator[CoverageRow]:
    """Yield each scenario's row as it finishes, so a caller can keep partial results.

    Every scenario draws from its own child stream of ``seed``.
    """
    streams = np.random.SeedSequence(seed).spawn(len(scenarios) + 1)
    geometry = fit_geometry(np.random.default_rng(streams[0]))
    for scenario, stream in zip(scenarios, streams[1:], strict=True):
        yield run_scenario(
            scenario,
            geometry,
            np.random.default_rng(stream),
            replicates=replicates,
            n_items=n_items,
            n_resamples=n_resamples,
            truth_items=truth_items,
        )


def to_frame(rows: Sequence[CoverageRow]) -> pd.DataFrame:
    """One row per scenario."""
    return pd.DataFrame.from_records(
        [
            {
                "scenario": row.scenario,
                "true_gap": row.true_gap,
                "replicates": row.replicates,
                "coverage": row.coverage,
                "above_zero": row.above_zero,
                "below_zero": row.below_zero,
                "mean_point": row.mean_point,
                "mean_width": row.mean_width,
                "unusable": row.unusable,
            }
            for row in rows
        ]
    )
