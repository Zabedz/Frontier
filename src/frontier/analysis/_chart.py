"""Shared chart chrome for the analysis figures: the ink and surface tokens, the
categorical hue order, and the atomic PNG writer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# A PNG commits to one theme, so every figure uses the light data-viz token set.
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
AXIS_INK = "#c3c2b7"
GRID = "#e1e0d9"
GAP_RED = "#e34948"

# Fixed light-surface order: a variant keeps its slot as the series set changes.
# Slots 2, 3, and 7 sit below 3:1 here, so the figures also carry direct labels.
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

# Two or more series always get a legend; the sub-3:1 slots above depend on it.
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
    # matplotlib takes the format from the extension, and ".png.tmp" is not one it knows.
    fig.savefig(tmp, format="png", facecolor=SURFACE, bbox_inches="tight")
    tmp.replace(out_path)
    return out_path
