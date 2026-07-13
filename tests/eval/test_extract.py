"""Confidence extraction and PriDe debiasing on the synthetic providers."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from synthetic import SentinelOracleProvider, StaticLogitProvider, sentinel_record

from frontier.eval.extract import score_items
from frontier.eval.provider import softmax
from frontier.eval.records import EvalRecord, letters_for


def test_confidence_from_hand_built_logits() -> None:
    row = np.full(8, -20.0)
    candidate_ids = [2, 3, 4, 5]
    row[2:6] = [2.0, 1.0, 0.0, 1.0]
    provider = StaticLogitProvider(row, candidate_ids)
    record = EvalRecord(
        qid="q", question="Q", options=("a", "b", "c", "d"), gold=0, subject="s", split="test"
    )
    out = score_items([record], provider, scheme="none")
    expected = softmax(np.array([2.0, 1.0, 0.0, 1.0]))
    assert out.probs[0, :4] == pytest.approx(expected)
    assert int(out.predicted[0]) == 0
    assert float(out.confidence[0]) == pytest.approx(float(expected.max()))
    assert out.robustness is None


def test_pride_recovers_gold_under_uniform_prior() -> None:
    gold = 2
    provider = SentinelOracleProvider(injected_prior=None, base_logit=2.0)
    out = score_items([sentinel_record(4, gold=gold)], provider, scheme="cyclic")
    assert int(out.predicted[0]) == gold
    assert out.robustness is not None
    assert out.robustness.letter_bias == pytest.approx(0.0, abs=1e-9)
    assert out.robustness.permutation_consistency == pytest.approx(1.0)
    assert out.robustness.debias_flip_rate == pytest.approx(0.0)


def test_pride_corrects_an_injected_letter_bias() -> None:
    gold = 2
    provider = SentinelOracleProvider(injected_prior=[0.7, 0.1, 0.1, 0.1], base_logit=1.0)
    record = sentinel_record(4, gold=gold)
    naive = score_items([record], provider, scheme="none")
    debiased = score_items([record], provider, scheme="cyclic")
    assert int(naive.predicted[0]) == 0  # the letter bias drags the canonical order onto A
    assert int(debiased.predicted[0]) == gold  # debiasing recovers the gold content
    assert debiased.robustness is not None
    assert debiased.robustness.debias_flip_rate == pytest.approx(1.0)


def test_letter_prior_recovers_injected_prior_and_bias_is_monotone() -> None:
    priors = [None, [0.4, 0.2, 0.2, 0.2], [0.55, 0.15, 0.15, 0.15], [0.7, 0.1, 0.1, 0.1]]
    records = [sentinel_record(4, gold=g, qid=f"q{g}") for g in range(4)]
    biases: list[float] = []
    prior_a: list[float] = []
    for injected in priors:
        provider = SentinelOracleProvider(injected_prior=injected, base_logit=2.0)
        out = score_items(records, provider, scheme="cyclic")
        assert out.robustness is not None
        expected = np.full(4, 0.25) if injected is None else np.asarray(injected)
        assert out.robustness.letter_prior == pytest.approx(expected)
        biases.append(out.robustness.letter_bias)
        prior_a.append(float(out.robustness.letter_prior[0]))
    assert all(earlier < later for earlier, later in itertools.pairwise(biases))
    assert all(earlier < later for earlier, later in itertools.pairwise(prior_a))
    assert biases[0] == pytest.approx(0.0, abs=1e-9)


def test_variable_option_count_scores_and_pads() -> None:
    provider = SentinelOracleProvider(injected_prior=None, base_logit=2.0)
    records = [sentinel_record(3, gold=1, qid="q3"), sentinel_record(5, gold=4, qid="q5")]
    out = score_items(records, provider, scheme="cyclic")
    assert out.probs.shape == (2, 5)
    assert out.n_options.tolist() == [3, 5]
    assert out.probs[0, 3:] == pytest.approx([0.0, 0.0])
    assert out.probs.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert out.predicted.tolist() == [1, 4]


def test_scheme_none_returns_canonical_order_and_no_robustness() -> None:
    provider = SentinelOracleProvider(injected_prior=None, base_logit=2.0)
    out = score_items([sentinel_record(4, gold=1)], provider, scheme="none")
    assert out.robustness is None
    assert out.probs[0] == pytest.approx(softmax(np.array([0.0, 2.0, 0.0, 0.0])))
    assert int(out.predicted[0]) == 1


def test_letters_for_rejects_degenerate_counts() -> None:
    with pytest.raises(ValueError, match="must be >= 2"):
        letters_for(1)
    with pytest.raises(ValueError, match="must be <= 26"):
        letters_for(27)


def test_argmax_tie_picks_lowest_index() -> None:
    row = np.full(6, -20.0)
    row[1:4] = [1.0, 1.0, 1.0]
    provider = StaticLogitProvider(row, [1, 2, 3])
    record = EvalRecord(
        qid="q", question="Q", options=("a", "b", "c"), gold=2, subject="s", split="test"
    )
    out = score_items([record], provider, scheme="none")
    assert int(out.predicted[0]) == 0


def test_score_items_rejects_empty_input() -> None:
    provider = SentinelOracleProvider()
    with pytest.raises(ValueError, match="no records"):
        score_items([], provider)
