"""The synthetic provider through the eval core into frontier.metrics."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import SentinelOracleProvider, StaticLogitProvider, sentinel_record

from frontier.eval.extract import exact_match, score_items, to_robustness, to_task_spec
from frontier.eval.records import EvalRecord
from frontier.metrics import calibration_report, check_predictions
from frontier.schema import EvalSpec, Robustness, TaskSpec


def _uniform_records(n: int) -> list[EvalRecord]:
    return [
        EvalRecord(
            qid=f"q{i}",
            question="Q?",
            options=("a", "b", "c", "d"),
            gold=i % 4,
            subject="sub",
            split="test",
        )
        for i in range(n)
    ]


def test_score_items_feeds_calibration_report() -> None:
    row = np.full(8, -20.0)
    row[0:4] = [1.5, 0.5, 0.5, 0.5]
    provider = StaticLogitProvider(row, [0, 1, 2, 3])
    records = _uniform_records(48)
    out = score_items(records, provider, scheme="none")

    report = calibration_report(out.probs, out.gold)
    assert np.isfinite(report.ece_equal_width)
    assert 0.0 <= report.ece_equal_width <= 1.0
    assert report.accuracy == pytest.approx(float(np.mean(exact_match(out.predicted, out.gold))))
    assert report.accuracy == pytest.approx(0.25)


def test_zero_padded_probs_pass_wp1_prediction_guard() -> None:
    provider = SentinelOracleProvider(injected_prior=None, base_logit=2.0)
    records = [
        sentinel_record(4, gold=1, qid="a"),
        sentinel_record(5, gold=4, qid="b"),
        sentinel_record(4, gold=0, qid="c"),
        sentinel_record(5, gold=2, qid="d"),
        sentinel_record(4, gold=3, qid="e"),
    ]
    out = score_items(records, provider, scheme="cyclic")
    assert out.probs.shape == (5, 5)
    check_predictions(out.probs, out.gold)
    report = calibration_report(out.probs, out.gold)
    assert 0.0 <= report.ece_equal_width <= 1.0


def test_to_task_spec_maps_eval_spec_onto_schema() -> None:
    spec = EvalSpec(
        task_name="mmlu",
        split="test",
        subset_size=2000,
        prompt_style="zeroshot",
        scoring="letter_softmax",
        permutation_scheme="cyclic",
        labels="redux",
        cot=False,
        seeds=(0,),
    )
    num_items = 123
    task = to_task_spec(spec, num_items=num_items)
    assert isinstance(task, TaskSpec)
    assert task.task_name == "mmlu"
    assert task.num_items == num_items
    assert task.permutation_scheme == "cyclic"
    assert task.labels == "redux"
    assert task.cot is False


def test_to_robustness_reduces_onto_schema_row() -> None:
    provider = SentinelOracleProvider(injected_prior=[0.6, 0.2, 0.1, 0.1], base_logit=2.0)
    out = score_items([sentinel_record(4, gold=2)], provider, scheme="cyclic")
    assert out.robustness is not None
    reduced = to_robustness(out.robustness)
    assert isinstance(reduced, Robustness)
    assert reduced.letter_bias == pytest.approx(out.robustness.letter_bias)
    assert reduced.permutation_consistency == pytest.approx(out.robustness.permutation_consistency)
    assert reduced.debias_flip_rate == pytest.approx(out.robustness.debias_flip_rate)
    assert to_robustness(None) is None
