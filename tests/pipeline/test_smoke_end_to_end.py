"""The CPU-smoke milestone under pytest: a real model, an in-memory slice.

Gated like the live backend test. The real ``HFLogitProvider`` scores a handful of
synthetic MMLU-shaped records (so only the model is live, not the dataset), and the
runner must land one schema-valid, fully-populated row in a ``tmp_path`` store.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from frontier.eval.records import EvalRecord
from frontier.io.store import ResultStore, read_rows
from frontier.pipeline.runner import run
from frontier.schema import EvalSpec

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
FP16 = CONFIG_ROOT / "variants" / "fp16.yaml"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("FRONTIER_LIVE_MODELS"),
        reason="live model download; set FRONTIER_LIVE_MODELS=1 to run",
    ),
]

_ITEMS = [
    ("The capital of France is", ("Paris", "Rome", "Berlin", "Madrid"), 0),
    ("Water is chemically", ("H2O", "CO2", "NaCl", "O2"), 0),
    ("The largest planet is", ("Mars", "Jupiter", "Venus", "Mercury"), 1),
    ("2 plus 2 equals", ("3", "4", "5", "6"), 1),
    ("The opposite of hot is", ("warm", "cold", "mild", "boiling"), 1),
    ("A triangle has how many sides", ("2", "3", "4", "5"), 1),
    ("The sun rises in the", ("west", "north", "east", "south"), 2),
    ("Ice is frozen", ("steam", "smoke", "water", "sand"), 2),
]


def _slice(spec: EvalSpec, *, seed: int) -> list[EvalRecord]:  # noqa: ARG001
    return [
        EvalRecord(
            qid=f"smoke:{i}",
            question=question,
            options=options,
            gold=gold,
            subject="smoke",
            split="test",
        )
        for i, (question, options, gold) in enumerate(_ITEMS)
    ]


def test_cpu_smoke_end_to_end(tmp_path: Path) -> None:
    import transformers  # noqa: PLC0415

    rows = run(
        FP16,
        mode="smoke",
        config_root=CONFIG_ROOT,
        results_root=tmp_path,
        slice_loader=_slice,
        timestamp="2026-07-13T00:00:00+00:00",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.backend.backend_version == transformers.__version__
    assert row.backend.inference_backend == "hf"
    assert row.robustness is not None
    assert 0.0 <= row.quality.accuracy <= 1.0
    assert np.isfinite(row.quality.ece_equal_width)
    assert row.task.num_items == len(_ITEMS)
    assert row.provenance.hardware_id.startswith("cpu:")

    # The real CPU latency rig fills the row: smoke pins batch [1], context [128].
    assert len(row.latency) == 1
    assert len(row.memory) == 1
    latency = row.latency[0]
    assert math.isfinite(latency.ttft_median_ms) and latency.ttft_median_ms > 0.0
    assert math.isfinite(latency.itl_median_ms) and latency.itl_median_ms > 0.0
    assert latency.machine_state is not None
    assert math.isfinite(row.tok_s_per_gb)

    assert len(read_rows(ResultStore(tmp_path))) == 1
