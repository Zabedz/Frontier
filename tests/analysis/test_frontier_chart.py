"""The frontier chart renders on every axis and grouping, and highlights the front.

The synthetic frame has four variants across three families; its memory-axis Pareto
front spans two families, the multi-family highlight the chart exists to show.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frontier.analysis.frontier_chart import ColorBy, frontier_chart
from frontier.analysis.load import XCost
from frontier.analysis.pareto import pareto_mask

_CROSS_TRACK_MATCH = "not comparable across tracks"

FRAME = pd.DataFrame(
    {
        "variant_name": ["fp16", "int8-weightonly", "int4-gptq", "distill-hardlabel"],
        "family": ["baseline", "ptq", "ptq", "distill"],
        "track": ["A", "A", "B", "A"],
        "accuracy": [0.85, 0.83, 0.80, 0.79],
        "accuracy_ci_low": [0.83, 0.81, 0.78, 0.77],
        "accuracy_ci_high": [0.87, 0.85, 0.82, 0.81],
        "itl_median_ms": [3.0, 3.2, 2.6, 3.1],
        "peak_vram_mb": [8000.0, 5000.0, 3500.0, 6000.0],
        "cost_inv": [5.0, 4.2, 3.0, 4.8],
    }
)

_AXES: tuple[XCost, ...] = ("memory", "latency", "cost_inv")
_GROUPINGS: tuple[ColorBy, ...] = ("family", "track")


def test_memory_front_spans_two_families() -> None:
    accuracy = np.asarray(FRAME["accuracy"], dtype=np.float64)
    memory = np.asarray(FRAME["peak_vram_mb"], dtype=np.float64)
    mask = pareto_mask(accuracy, memory)
    on_front = set(FRAME.loc[mask, "variant_name"].tolist())
    assert on_front == {"fp16", "int8-weightonly", "int4-gptq"}
    front_families = set(FRAME.loc[mask, "family"].tolist())
    assert front_families == {"baseline", "ptq"}


def test_every_axis_and_grouping_renders_a_nonempty_png(tmp_path: Path) -> None:
    for x in _AXES:
        for color_by in _GROUPINGS:
            out = tmp_path / f"frontier-{x}-{color_by}.png"
            result = frontier_chart(
                FRAME, x=x, color_by=color_by, out_path=out, allow_cross_track=True
            )
            assert result == out
            assert out.stat().st_size > 0


def test_annotate_all_also_renders(tmp_path: Path) -> None:
    out = tmp_path / "frontier-all.png"
    frontier_chart(FRAME, x="memory", out_path=out, annotate="all")
    assert out.stat().st_size > 0


def test_non_cross_track_axis_warns_on_a_multi_track_frame(tmp_path: Path) -> None:
    out = tmp_path / "latency.png"
    with pytest.warns(UserWarning, match=_CROSS_TRACK_MATCH):
        frontier_chart(FRAME, x="latency", out_path=out)
    assert out.stat().st_size > 0  # the guard warns but still draws


def test_cross_track_opt_in_silences_the_warning(tmp_path: Path) -> None:
    out = tmp_path / "latency.png"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frontier_chart(FRAME, x="latency", out_path=out, allow_cross_track=True)
    assert not any(_CROSS_TRACK_MATCH in str(entry.message) for entry in caught)


def test_memory_axis_never_warns_across_tracks(tmp_path: Path) -> None:
    out = tmp_path / "memory.png"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frontier_chart(FRAME, x="memory", out_path=out)
    assert not any(_CROSS_TRACK_MATCH in str(entry.message) for entry in caught)
