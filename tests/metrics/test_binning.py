"""Binning layer: edges, the histogram convention, and the degenerate collapse."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_softmax

from frontier.metrics.binning import bin_edges, bin_stats


def test_equal_width_edges_are_linspace() -> None:
    confidence = make_softmax(200, 4, np.random.default_rng(0)).max(axis=1)
    edges = bin_edges(confidence, 10, "equal_width")
    assert np.array_equal(edges, np.linspace(0.0, 1.0, 11))


def test_equal_mass_edges_span_the_unit_interval_and_increase() -> None:
    confidence = make_softmax(500, 4, np.random.default_rng(1)).max(axis=1)
    edges = bin_edges(confidence, 10, "equal_mass")
    assert edges[0] == 0.0
    assert edges[-1] == 1.0
    assert bool(np.all(np.diff(edges) > 0.0))


def test_bin_stats_zeroes_empty_bins_and_counts_sum_to_n() -> None:
    rng = np.random.default_rng(2)
    confidence = make_softmax(1000, 5, rng).max(axis=1)
    correct = rng.uniform(0.0, 1.0, size=1000) < confidence
    stats = bin_stats(confidence, correct, 10, "equal_width")
    empty = stats.count == 0
    assert bool(empty.any())  # a max over 5 classes never lands below 0.2
    assert bool(np.all(stats.mean_confidence[empty] == 0.0))
    assert bool(np.all(stats.accuracy[empty] == 0.0))
    assert int(stats.count.sum()) == confidence.shape[0]


def test_interior_edge_lands_in_the_upper_bin() -> None:
    edges = np.linspace(0.0, 1.0, 11)
    interior = 3
    confidence = np.array([edges[interior]])
    correct = np.array([True])
    stats = bin_stats(confidence, correct, 10, "equal_width")
    assert int(stats.count[interior]) == 1
    assert int(stats.count.sum()) == 1


def test_confidence_of_one_lands_in_the_last_bin() -> None:
    confidence = np.array([1.0])
    correct = np.array([True])
    stats = bin_stats(confidence, correct, 10, "equal_width")
    assert int(stats.count[-1]) == 1
    assert int(stats.count.sum()) == 1


def test_degenerate_equal_mass_dedupes_without_raising() -> None:
    confidence = np.full(100, 0.7)
    correct = np.zeros(100, dtype=np.bool_)
    edges = bin_edges(confidence, 10, "equal_mass")
    assert bool(np.all(np.diff(edges) > 0.0))
    stats = bin_stats(confidence, correct, 10, "equal_mass")
    assert int(stats.count.sum()) == confidence.shape[0]


def test_bin_edges_rejects_non_positive_bin_count() -> None:
    confidence = np.array([0.3, 0.6, 0.9])
    with pytest.raises(ValueError, match="n_bins"):
        bin_edges(confidence, 0, "equal_width")
