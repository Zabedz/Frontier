"""Read-and-plot layer over the append-only result store.

Turns stored ``ResultRow``s (and their per-item predictions sidecars) into the two
first-class figures: the accuracy-versus-cost frontier chart with its Pareto front,
and the reliability-diagram gallery with the ECE-vs-bin-count sweep. Nothing here
recomputes a model output; it reads ``results/`` and writes ``plots/``.
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

__all__ = [
    "DEFAULT_SWEEP_BINS",
    "X_AXES",
    "XAxisSpec",
    "XCost",
    "collapse_seeds",
    "draw_reliability",
    "ece_bins_curve",
    "ece_bins_sweep_figure",
    "frontier_chart",
    "load_all_predictions",
    "load_predictions_for_variant",
    "load_tidy",
    "pareto_mask",
    "pareto_order",
    "prediction_labels",
    "reliability_figure",
    "reliability_gallery",
]
