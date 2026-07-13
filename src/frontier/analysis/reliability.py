"""Reliability diagrams and the ECE-vs-bin-count sweep, all read off the sidecars.

Every curve and every ECE comes from the committed ``frontier.metrics`` core; this
module adds no binning maths, only the drawing. The sweep is the Option-A payoff: it
recomputes ECE at any bin count directly from the per-item ``(confidence, correct)``
arrays, so it is free to go finer than the six bin counts frozen on the stored row.

The matplotlib ``Figure``/``Axes`` objects are an untyped boundary (a skipped import
for mypy), so the drawing helpers take them as ``Any``; the module returns concrete
``Path`` and ``dict`` values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from frontier.analysis._chart import (
    AXIS_INK,
    CATEGORICAL,
    GAP_RED,
    GRID,
    LEGEND_MIN_SERIES,
    MUTED_INK,
    PRIMARY_INK,
    SECONDARY_INK,
    SURFACE,
    atomic_save,
)
from frontier.io.predictions import PredictionRows
from frontier.metrics.binning import BinScheme
from frontier.metrics.calibration import (
    DEFAULT_BINS,
    ReliabilityCurve,
    ece_from_confidence,
    reliability_from_confidence,
)

PRIMARY_HUE = CATEGORICAL[0]
_GAP_ALPHA = 0.35
_BAR_FRACTION = 0.9
_GAP_FRACTION = 0.55

# A sweep deliberately finer than the six bin counts frozen on the stored row, so the
# figure demonstrates that the sidecar re-bins freely (the Option-A payoff).
DEFAULT_SWEEP_BINS: tuple[int, ...] = (2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50)


def ece_bins_curve(
    preds: PredictionRows,
    *,
    bin_counts: Sequence[int] = DEFAULT_SWEEP_BINS,
    scheme: BinScheme = "equal_width",
) -> dict[int, float]:
    """Map each bin count to its ECE, recomputed from the pooled per-item arrays."""
    return {
        n: ece_from_confidence(preds.confidence, preds.correct, n_bins=n, scheme=scheme)
        for n in bin_counts
    }


def draw_reliability(
    ax: Any,
    curve: ReliabilityCurve,
    *,
    color: str,
    title: str | None = None,
    ece: float | None = None,
) -> None:
    """Draw one Guo-style reliability diagram onto an existing ``Axes``.

    The per-bin accuracy is drawn as bars at the bin centres; the calibration gap
    (over- or under-confidence) is overlaid as a translucent red bar from accuracy to
    mean confidence. Empty bins carry ``np.nan`` and are dropped, not drawn at the
    origin.
    """
    edges = curve.edges
    centres = (edges[:-1] + edges[1:]) / 2.0
    widths = edges[1:] - edges[:-1]
    populated = ~np.isnan(curve.accuracy)
    accuracy = curve.accuracy[populated]
    mean_confidence = curve.mean_confidence[populated]
    centres = centres[populated]
    widths = widths[populated]

    ax.plot([0.0, 1.0], [0.0, 1.0], color=MUTED_INK, linewidth=1.0, zorder=1)
    ax.bar(
        centres,
        accuracy,
        width=widths * _BAR_FRACTION,
        color=color,
        zorder=3,
        align="center",
    )
    gap_bottom = np.minimum(accuracy, mean_confidence)
    gap_height = np.abs(mean_confidence - accuracy)
    ax.bar(
        centres,
        gap_height,
        bottom=gap_bottom,
        width=widths * _GAP_FRACTION,
        color=GAP_RED,
        alpha=_GAP_ALPHA,
        zorder=4,
        align="center",
    )
    if title is not None:
        ax.set_title(title, color=PRIMARY_INK, fontsize=10, loc="left")
    if ece is not None:
        ax.text(
            0.04,
            0.94,
            f"ECE {ece:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=SECONDARY_INK,
        )


def reliability_figure(
    preds: PredictionRows,
    *,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
    out_path: Path,
    title: str | None = None,
    color: str = PRIMARY_HUE,
) -> Path:
    """Render a single-variant reliability diagram to ``out_path`` and return the path."""
    curve = reliability_from_confidence(preds.confidence, preds.correct, n_bins, scheme)
    value = ece_from_confidence(preds.confidence, preds.correct, n_bins=n_bins, scheme=scheme)

    fig = Figure(figsize=(4.5, 4.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.subplots()
    _style_cell(ax)
    draw_reliability(ax, curve, color=color, title=title, ece=value)
    ax.set_xlabel("Confidence", color=PRIMARY_INK, fontsize=10)
    ax.set_ylabel("Accuracy", color=PRIMARY_INK, fontsize=10)
    return atomic_save(fig, Path(out_path))


def reliability_gallery(
    preds_by_variant: Mapping[str, PredictionRows],
    *,
    order: Sequence[str] | None = None,
    n_bins: int = DEFAULT_BINS,
    scheme: BinScheme = "equal_width",
    ncols: int = 4,
    out_path: Path,
    title: str | None = None,
) -> Path:
    """Render a small-multiples gallery, one cell per variant, to ``out_path``.

    Cells share 0..1 axes so the shape-against-diagonal reading is comparable. ``order``
    fixes the cell sequence; it defaults to the sorted variant names (a caller holding
    family and bit-width metadata can pass a bit-width-then-family order). All cells use
    one hue: the facet position and cell title carry identity, not colour.
    """
    names = list(order) if order is not None else sorted(preds_by_variant)
    count = len(names)
    nrows = max(1, math.ceil(count / ncols))

    fig = Figure(figsize=(ncols * 2.7, nrows * 2.7), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for index in range(nrows * ncols):
        row, col = divmod(index, ncols)
        ax = axes[row][col]
        if index >= count:
            ax.axis("off")
            continue
        name = names[index]
        preds = preds_by_variant[name]
        curve = reliability_from_confidence(preds.confidence, preds.correct, n_bins, scheme)
        value = ece_from_confidence(preds.confidence, preds.correct, n_bins=n_bins, scheme=scheme)
        _style_cell(ax)
        draw_reliability(ax, curve, color=PRIMARY_HUE, title=name, ece=value)

    fig.supxlabel("Confidence", color=PRIMARY_INK, fontsize=11)
    fig.supylabel("Accuracy", color=PRIMARY_INK, fontsize=11)
    if title is not None:
        fig.suptitle(title, color=PRIMARY_INK, fontsize=13)
    return atomic_save(fig, Path(out_path))


def ece_bins_sweep_figure(
    preds_by_variant: Mapping[str, PredictionRows],
    *,
    bin_counts: Sequence[int] = DEFAULT_SWEEP_BINS,
    scheme: BinScheme = "equal_width",
    out_path: Path,
    title: str | None = None,
) -> Path:
    """Render one ECE-vs-bin-count line per variant to ``out_path`` and return the path.

    This is the figure that enforces "never publish a single ECE": ECE is recomputed
    from the sidecars at every bin count in ``bin_counts``, free to go finer than the
    stored sweep. Series are coloured in the palette's fixed order and a legend is
    always present.
    """
    names = sorted(preds_by_variant)
    xs = list(bin_counts)

    fig = Figure(figsize=(7.0, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.subplots()
    _style_line_axes(ax)
    for index, name in enumerate(names):
        curve = ece_bins_curve(preds_by_variant[name], bin_counts=bin_counts, scheme=scheme)
        ys = [curve[n] for n in xs]
        ax.plot(
            xs,
            ys,
            color=_series_color(index),
            linewidth=2.0,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=name,
            zorder=3,
        )
    ax.set_xlabel("Number of bins", color=PRIMARY_INK, fontsize=10)
    ax.set_ylabel("ECE", color=PRIMARY_INK, fontsize=10)
    ax.set_ylim(bottom=0.0)
    if len(names) >= LEGEND_MIN_SERIES:
        ax.legend(frameon=False, fontsize=8, labelcolor=PRIMARY_INK)
    if title is not None:
        ax.set_title(title, color=PRIMARY_INK, fontsize=12, loc="left")
    return atomic_save(fig, Path(out_path))


def _style_cell(ax: Any) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.grid(visible=True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED_INK, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_INK)


def _style_line_axes(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(visible=True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED_INK, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_INK)


def _series_color(index: int) -> str:
    if index < len(CATEGORICAL):
        return CATEGORICAL[index]
    return MUTED_INK
