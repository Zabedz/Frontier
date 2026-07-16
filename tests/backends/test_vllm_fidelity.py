"""The FP16 HF-vs-vLLM fidelity gate. Pod-only, gated, skipped on the CPU loop.

Scores an unquantised FP16 Qwen2.5-3B through the HF backend and the vLLM backend on the
same MMLU slice and asserts the candidate softmax and the ECE agree. This is the
architecture rule that a Track-B method reproduces the Track-A baseline before its ECE is
admitted next to an HF ECE, and it is the empirical proof that the ``logprobs=-1``
extraction reproduces the HF candidate softmax by construction.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("vllm")
pytest.importorskip("transformers")
pytest.importorskip("torch")

from frontier.backends.hf import HFLogitProvider
from frontier.backends.vllm import VllmLogitProvider
from frontier.eval.extract import exact_match, score_items
from frontier.eval.records import EvalRecord
from frontier.metrics.report import calibration_report

QWEN = "Qwen/Qwen2.5-3B-Instruct"
PROB_TOL = 5e-2
ECE_TOL = 1e-2

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not os.environ.get("FRONTIER_LIVE_MODELS"),
        reason="live GPU model; set FRONTIER_LIVE_MODELS=1 on the pod to run",
    ),
]

_ITEMS = [
    ("The capital of France is", ("Paris", "Rome", "Berlin", "Madrid"), 0),
    ("Water is chemically", ("H2O", "CO2", "NaCl", "O2"), 0),
    ("The largest planet is", ("Mars", "Jupiter", "Venus", "Mercury"), 1),
    ("2 plus 2 equals", ("3", "4", "5", "6"), 1),
    ("The opposite of hot is", ("warm", "cold", "mild", "boiling"), 1),
    ("The sun rises in the", ("west", "north", "east", "south"), 2),
]


def _records() -> list[EvalRecord]:
    return [
        EvalRecord(qid=f"g:{i}", question=q, options=o, gold=g, subject="gate", split="test")
        for i, (q, o, g) in enumerate(_ITEMS)
    ]


def test_fp16_vllm_reproduces_hf_candidate_softmax_and_ece() -> None:
    records = _records()
    hf = HFLogitProvider(model_id=QWEN, device="cuda", weight_dtype="fp16")
    vllm = VllmLogitProvider(model=QWEN, tokenizer_id=QWEN, device="cuda", weight_dtype="fp16")

    hf_out = score_items(records, hf, scheme="cyclic")
    vllm_out = score_items(records, vllm, scheme="cyclic")

    assert np.max(np.abs(hf_out.confidence - vllm_out.confidence)) < PROB_TOL

    hf_ece = calibration_report(hf_out.probs, hf_out.gold).ece_equal_width
    vllm_ece = calibration_report(vllm_out.probs, vllm_out.gold).ece_equal_width
    assert abs(hf_ece - vllm_ece) < ECE_TOL

    hf_acc = exact_match(hf_out.predicted, hf_out.gold)
    vllm_acc = exact_match(vllm_out.predicted, vllm_out.gold)
    assert np.array_equal(hf_acc, vllm_acc)
