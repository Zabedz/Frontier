"""Candidate-id resolution and the two letter primitives, on hand values."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import FakeTokenizer

from frontier.eval.provider import letter_probs, resolve_candidate_ids, softmax


def test_resolve_candidate_ids_picks_leading_space_variant() -> None:
    tokenizer = FakeTokenizer(
        {" A": [10], " B": [11], " C": [12], " D": [13], "A": [99], "B": [98], "C": [97], "D": [96]}
    )
    with_space = resolve_candidate_ids(tokenizer, "ABCD", leading_space=True)
    without_space = resolve_candidate_ids(tokenizer, "ABCD", leading_space=False)
    assert with_space.tolist() == [10, 11, 12, 13]
    assert without_space.tolist() == [99, 98, 97, 96]
    assert with_space.dtype == np.intp


def test_resolve_candidate_ids_rejects_multi_token_letter() -> None:
    tokenizer = FakeTokenizer({" A": [10, 11]})
    with pytest.raises(ValueError, match=r"'A'.*not a single token.*\[10, 11\]"):
        resolve_candidate_ids(tokenizer, "A")


def test_softmax_matches_hand_value_and_normalises() -> None:
    result = softmax(np.array([0.0, np.log(3.0)]))
    assert result == pytest.approx([0.25, 0.75])
    assert float(result.sum()) == pytest.approx(1.0)


def test_letter_probs_gathers_candidates_then_softmaxes() -> None:
    logits = np.array([2.0, 9.0, 1.0, 0.0, 9.0, 1.0])
    candidate_ids = np.array([0, 2, 3, 5], dtype=np.intp)
    probs = letter_probs(logits, candidate_ids)
    assert probs == pytest.approx(softmax(np.array([2.0, 1.0, 0.0, 1.0])))
    assert int(np.argmax(probs)) == 0
