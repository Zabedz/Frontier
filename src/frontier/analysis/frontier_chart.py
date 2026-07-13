"""The frontier chart: accuracy on y, a cost axis on x, one marker per variant.

The desirable corner is upper-left (high accuracy, low cost), so all three cost axes
read low-is-better. The Pareto front is traced with a connecting envelope and its
markers are emphasized; dominated variants stay as plain markers. Colour is assigned
by the entity (family or track) in the palette's fixed order, never by rank, and the
front variants are directly labelled so identity never rests on colour alone.

The matplotlib ``Figure``/``Axes`` objects are an untyped boundary (the package is a
skipped import for mypy), so the drawing helpers take them as ``Any``; every value
this module returns stays a concrete ``Path``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

from frontier.analysis._chart import (
    AXIS_INK,
    FAMILY_COLORS,
    FAMILY_LABELS,
    GRID,
    LEGEND_MIN_SERIES,
    MUTED_INK,
    PRIMARY_INK,
    SECONDARY_INK,
    SURFACE,
    TRACK_COLORS,
    atomic_save,
)
from frontier.analysis.load import X_AXES, XAxisSpec, XCost
from frontier.analysis.pareto import pareto_mask, pareto_order

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

ColorBy = Literal["family", "track"]
Annotate = Literal["front", "all", "none"]

_MARKER_AREA = 90.0
_FRONT_RING_AREA = 190.0
_Y_HEADROOM = 0.03
_MIN_FRONT_POINTS = 2


def frontier_chart(
    tidy: pd.DataFrame,
    *,
    x: XCost = "memory",
    color_by: ColorBy = "family",
    out_path: Path,
    title: str | None = None,
    annotate: Annotate = "front",
    allow_cross_track: bool = False,
) -> Path:
    """Render the frontier chart to ``out_path`` (PNG) and return the path.

    ``tidy`` is a per-variant frame (one row per variant, e.g. from ``collapse_seeds``).
    ``x`` selects the cost axis via ``X_AXES``; the mixed-track default is memory, the
    axis comparable across both tracks. ``color_by`` colours markers by family or track.
    ``annotate`` labels the Pareto front (``"front"``), every variant (``"all"``), or
    none. Drawing a non-cross-track axis (latency, cost_inv) on a frame that spans both
    tracks warns, because those axes embed a per-backend clock and are not comparable
    across tracks; ``allow_cross_track=True`` acknowledges the caveat and silences it.
    """
    spec = X_AXES[x]
    _warn_if_cross_track(tidy, spec, allow_cross_track=allow_cross_track)
    accuracy = tidy["accuracy"].to_numpy(dtype=np.float64)
    cost = tidy[spec.column].to_numpy(dtype=np.float64)
    ci_low = tidy["accuracy_ci_low"].to_numpy(dtype=np.float64)
    ci_high = tidy["accuracy_ci_high"].to_numpy(dtype=np.float64)
    names = [str(name) for name in tidy["variant_name"].tolist()]
    groups = [str(group) for group in tidy[color_by].tolist()]

    finite = np.isfinite(accuracy) & np.isfinite(cost)
    front = pareto_mask(accuracy, cost)

    fig = Figure(figsize=(7.0, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    ax.grid(visible=True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)

    _draw_error_bars(ax, cost, accuracy, ci_low, ci_high, finite)
    _draw_front_line(ax, cost, accuracy)
    drawn_groups = _draw_markers(ax, cost, accuracy, groups, finite, color_by)
    _emphasize_front(ax, cost, accuracy, front)
    _draw_labels(ax, cost, accuracy, names, front, finite, annotate)

    _style_axes(ax, spec.label)
    if drawn_groups >= LEGEND_MIN_SERIES:
        ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=PRIMARY_INK)
    if title is not None:
        ax.set_title(title, color=PRIMARY_INK, fontsize=12, loc="left")

    return atomic_save(fig, Path(out_path))


def _draw_error_bars(
    ax: Any,
    cost: FloatArray,
    accuracy: FloatArray,
    ci_low: FloatArray,
    ci_high: FloatArray,
    finite: BoolArray,
) -> None:
    if not finite.any():
        return
    lower = np.clip(accuracy - ci_low, 0.0, None)
    upper = np.clip(ci_high - accuracy, 0.0, None)
    yerr = np.vstack([lower[finite], upper[finite]])
    ax.errorbar(
        cost[finite],
        accuracy[finite],
        yerr=yerr,
        fmt="none",
        ecolor=MUTED_INK,
        elinewidth=1.0,
        capsize=2.5,
        capthick=1.0,
        zorder=2,
    )


def _draw_front_line(ax: Any, cost: FloatArray, accuracy: FloatArray) -> None:
    order = pareto_order(accuracy, cost)
    if order.size < _MIN_FRONT_POINTS:
        return
    ax.plot(
        cost[order],
        accuracy[order],
        color=SECONDARY_INK,
        linewidth=2.0,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=3,
    )


def _draw_markers(
    ax: Any,
    cost: FloatArray,
    accuracy: FloatArray,
    groups: list[str],
    finite: BoolArray,
    color_by: ColorBy,
) -> int:
    palette = FAMILY_COLORS if color_by == "family" else TRACK_COLORS
    group_array = np.array(groups, dtype=object)
    drawn = 0
    for key, hue in palette.items():
        selected = finite & (group_array == key)
        if not selected.any():
            continue
        ax.scatter(
            cost[selected],
            accuracy[selected],
            s=_MARKER_AREA,
            c=hue,
            edgecolors=SURFACE,
            linewidths=2.0,
            zorder=4,
            label=_group_label(color_by, key),
        )
        drawn += 1
    return drawn


def _emphasize_front(ax: Any, cost: FloatArray, accuracy: FloatArray, front: BoolArray) -> None:
    if not front.any():
        return
    ax.scatter(
        cost[front],
        accuracy[front],
        s=_FRONT_RING_AREA,
        facecolors="none",
        edgecolors=SECONDARY_INK,
        linewidths=1.5,
        zorder=5,
    )


def _draw_labels(
    ax: Any,
    cost: FloatArray,
    accuracy: FloatArray,
    names: list[str],
    front: BoolArray,
    finite: BoolArray,
    annotate: Annotate,
) -> None:
    if annotate == "front":
        indices = np.flatnonzero(front)
    elif annotate == "all":
        indices = np.flatnonzero(finite)
    else:
        return
    for i in indices:
        ax.annotate(
            names[i],
            (cost[i], accuracy[i]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            color=SECONDARY_INK,
        )


def _style_axes(ax: Any, x_label: str) -> None:
    ax.set_xlabel(x_label, color=PRIMARY_INK, fontsize=10)
    ax.set_ylabel("Accuracy", color=PRIMARY_INK, fontsize=10)
    ax.set_ylim(0.0, 1.0 + _Y_HEADROOM)
    ax.tick_params(colors=MUTED_INK, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_INK)


def _group_label(color_by: ColorBy, key: str) -> str:
    if color_by == "track":
        return f"Track {key}"
    return FAMILY_LABELS.get(key, key)


def _warn_if_cross_track(tidy: pd.DataFrame, spec: XAxisSpec, *, allow_cross_track: bool) -> None:
    if spec.cross_track or allow_cross_track or "track" not in tidy.columns:
        return
    tracks = int(tidy["track"].nunique())
    if tracks > 1:
        warnings.warn(
            f"axis {spec.key!r} is not comparable across tracks, but the frame spans "
            f"{tracks} tracks: a latency or throughput number from one backend cannot sit "
            "in the same column as another's. Facet by track, use x='memory', or pass "
            "allow_cross_track=True to acknowledge the caveat.",
            stacklevel=2,
        )
