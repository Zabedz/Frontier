"""Shared chart chrome for the analysis figures: the fixed ink and surface tokens,
the categorical hue order, and the atomic PNG writer.

A PNG commits to one theme, so the figures render on the project's light data-viz
surface and draw from the light-mode token set. ``atomic_save`` writes to a sibling
``.tmp`` and renames, so a reader never observes a half-written PNG. matplotlib infers
its format from the filename extension, and a ``.png.tmp`` name is not a known format,
so the writer passes ``format="png"`` explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
AXIS_INK = "#c3c2b7"
GRID = "#e1e0d9"
GAP_RED = "#e34948"

# The palette's categorical hues in their fixed light-surface order. Assigned to an
# entity in a stable order and never cycled: a variant keeps its slot as the series
# set changes. Slots 2, 3, and 7 sit below 3:1 on the light surface, so identity never
# rests on colour alone (direct labels and a legend are always present).
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",
    "#1baf7a",
    "#eda100",
    "#008300",
    "#4a3aa7",
    "#e34948",
    "#e87ba4",
    "#eb6834",
)

# The data-viz rule: a legend is present once two or more series share the axes.
LEGEND_MIN_SERIES = 2

FAMILY_COLORS: dict[str, str] = {
    "baseline": "#2a78d6",
    "ptq": "#1baf7a",
    "qat": "#eda100",
    "distill": "#008300",
}
FAMILY_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "ptq": "PTQ",
    "qat": "QAT",
    "distill": "Distillation",
}
TRACK_COLORS: dict[str, str] = {"A": "#2a78d6", "B": "#1baf7a"}


def atomic_save(fig: Any, out_path: Path) -> Path:
    """Save ``fig`` to ``out_path`` (PNG) via a sibling ``.tmp`` and an atomic rename."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.name}.tmp")
    fig.savefig(tmp, format="png", facecolor=SURFACE, bbox_inches="tight")
    tmp.replace(out_path)
    return out_path
