"""Read-and-plot layer over the result store: reads ``results/``, writes ``plots/``.

Figures come from stored ``ResultRow``s and their per-item sidecars; no model output is
recomputed here.
"""

from __future__ import annotations

from frontier.analysis.frontier_chart import frontier_chart
from frontier.analysis.load import (
    X_AXES,
    XAxisSpec,
    XCost,
    collapse_seeds,
    load_all_predictions,
    load_predictions_for_variant,
    load_tidy,
    prediction_labels,
)
from frontier.analysis.pareto import pareto_mask, pareto_order
from frontier.analysis.reliability import (
    DEFAULT_SWEEP_BINS,
    draw_reliability,
    ece_bins_curve,
    ece_bins_sweep_figure,
    reliability_figure,
    reliability_gallery,
)
from frontier.analysis.significance import (
    DEFAULT_REFERENCES_PATH,
    PairSignificance,
    Skipped,
    VariantPair,
    load_references,
    pair_significance,
    resolve_pairs,
    significance_table,
    to_frame,
)

__all__ = [
    "DEFAULT_REFERENCES_PATH",
    "DEFAULT_SWEEP_BINS",
    "X_AXES",
    "PairSignificance",
    "Skipped",
    "VariantPair",
    "XAxisSpec",
    "XCost",
    "collapse_seeds",
    "draw_reliability",
    "ece_bins_curve",
    "ece_bins_sweep_figure",
    "frontier_chart",
    "load_all_predictions",
    "load_predictions_for_variant",
    "load_references",
    "load_tidy",
    "pair_significance",
    "pareto_mask",
    "pareto_order",
    "prediction_labels",
    "reliability_figure",
    "reliability_gallery",
    "resolve_pairs",
    "significance_table",
    "to_frame",
]
