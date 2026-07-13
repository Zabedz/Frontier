"""Live loader checks against the Hub. Excluded from the default run.

Marked ``slow`` and skipped unless ``FRONTIER_LIVE_DATASETS`` is set, so the default
``uv run pytest`` never touches the network. When enabled it confirms the three
datasets still load natively under ``datasets`` 4.6 (no loading script) and that the
normalisers produce in-range golds, the one thing the inline fixtures cannot prove.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("datasets")

from frontier.eval.loaders import load_arc_challenge, load_mmlu, load_mmlu_redux
from frontier.eval.records import EvalRecord

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("FRONTIER_LIVE_DATASETS"),
        reason="live dataset download; set FRONTIER_LIVE_DATASETS=1 to run",
    ),
]


def _assert_in_range(records: list[EvalRecord]) -> None:
    assert records
    for record in records:
        assert record.options
        assert 0 <= record.gold < len(record.options)


def test_live_mmlu_loads_and_normalises() -> None:
    _assert_in_range(load_mmlu(split="test", subset=5, seed=0))


def test_live_arc_loads_and_normalises() -> None:
    _assert_in_range(load_arc_challenge(split="test", subset=5, seed=0))


def test_live_mmlu_redux_loads_and_normalises() -> None:
    records = load_mmlu_redux(policy="clean_subset", subjects=["anatomy"], subset=5, seed=0)
    _assert_in_range(records)
    assert all(record.error_type is not None for record in records)
