"""Reliability figure, gallery, and ECE-vs-bins sweep render to non-empty PNGs.

The sweep recomputes ECE from the sidecar at more bin counts than the row freezes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from frontier.analysis.reliability import (
    DEFAULT_SWEEP_BINS,
    ece_bins_curve,
    ece_bins_sweep_figure,
    reliability_figure,
    reliability_gallery,
)
from frontier.io.predictions import PredictionRows
from frontier.metrics import DEFAULT_SWEEP


def _preds(seed: int, n: int, low: float) -> PredictionRows:
    rng = np.random.default_rng(seed)
    confidence = rng.uniform(low, 1.0, size=n)
    correct = rng.uniform(0.0, 1.0, size=n) < confidence
    gold = rng.integers(0, 4, size=n).astype(np.intp)
    predicted = np.where(correct, gold, (gold + 1) % 4).astype(np.intp)
    return PredictionRows(confidence, correct.astype(np.bool_), gold, predicted, options=None)


def test_reliability_figure_renders(tmp_path: Path) -> None:
    out = reliability_figure(_preds(0, 600, 0.4), out_path=tmp_path / "reliability.png")
    assert out.stat().st_size > 0


def test_gallery_renders_across_variants_of_different_bit_widths(tmp_path: Path) -> None:
    preds = {
        "fp16": _preds(0, 600, 0.5),
        "int4-gptq": _preds(1, 600, 0.35),
        "qat-3bit-lora": _preds(2, 600, 0.3),
    }
    out = reliability_gallery(preds, out_path=tmp_path / "gallery.png")
    assert out.stat().st_size > 0


def test_sweep_renders_and_rebins_finer_than_the_stored_points(tmp_path: Path) -> None:
    preds = {"fp16": _preds(0, 600, 0.5), "int4-gptq": _preds(1, 600, 0.35)}
    out = ece_bins_sweep_figure(preds, out_path=tmp_path / "sweep.png")
    assert out.stat().st_size > 0

    curve = ece_bins_curve(preds["fp16"])
    assert len(curve) == len(DEFAULT_SWEEP_BINS)
    assert len(curve) > len(DEFAULT_SWEEP)
