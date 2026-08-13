"""Synthetic logit providers and inline records for the eval suite; no model is loaded.

The sentinel provider's log-prior cancels in cyclic aggregation, so PriDe is exactly
invertible and the tests can assert exact recovery.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from frontier.eval.records import LETTERS, EvalRecord

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

SENTINEL = "<<GOLD>>"
VOCAB = 200
NON_CANDIDATE_LOGIT = -30.0
_LETTER_ID = {letter: 100 + i for i, letter in enumerate(LETTERS)}
_LETTER_INDEX = {letter: i for i, letter in enumerate(LETTERS)}


def sentinel_record(
    n_options: int, gold: int, *, qid: str = "q", subject: str = "sub"
) -> EvalRecord:
    """A record whose gold option carries the sentinel, so the oracle can read it."""
    options = tuple(
        f"option {i} {SENTINEL}" if i == gold else f"option {i}" for i in range(n_options)
    )
    return EvalRecord(
        qid=qid, question="Q?", options=options, gold=gold, subject=subject, split="test"
    )


class SentinelOracleProvider:
    """Logits that place a fixed preference on the sentinel-marked option.

    Each candidate logit is ``base_logit`` at the sentinel position plus
    ``log(injected_prior[position])``, so the softmax gives preference times prior.
    """

    def __init__(
        self, *, injected_prior: Sequence[float] | None = None, base_logit: float = 3.0
    ) -> None:
        self._prior = (
            None if injected_prior is None else np.asarray(injected_prior, dtype=np.float64)
        )
        self._base_logit = base_logit

    def candidate_token_ids(self, letters: Sequence[str]) -> IntArray:
        return np.asarray([_LETTER_ID[letter] for letter in letters], dtype=np.intp)

    def next_token_logits(self, prompts: Sequence[str]) -> FloatArray:
        return np.stack([self._logits_for(prompt) for prompt in prompts])

    def _prior_for(self, n: int) -> FloatArray:
        if self._prior is None:
            return np.full(n, 1.0 / n, dtype=np.float64)
        if len(self._prior) != n:
            raise ValueError(
                f"injected_prior has length {len(self._prior)} but item has {n} options"
            )
        return self._prior

    def _logits_for(self, prompt: str) -> FloatArray:
        n, gold_position = _parse_option_lines(prompt)
        prior = self._prior_for(n)
        row = np.full(VOCAB, NON_CANDIDATE_LOGIT, dtype=np.float64)
        for position in range(n):
            preference = self._base_logit if position == gold_position else 0.0
            row[_LETTER_ID[LETTERS[position]]] = preference + float(np.log(prior[position]))
        return row


class StaticLogitProvider:
    """Returns one caller-supplied logit row for every prompt, with a fixed id map."""

    def __init__(
        self, logit_row: FloatArray | Sequence[float], candidate_ids: Sequence[int]
    ) -> None:
        self._row = np.asarray(logit_row, dtype=np.float64)
        self._ids = np.asarray(candidate_ids, dtype=np.intp)

    def candidate_token_ids(self, letters: Sequence[str]) -> IntArray:
        if len(letters) != len(self._ids):
            raise ValueError(f"provider holds {len(self._ids)} ids but item needs {len(letters)}")
        return self._ids

    def next_token_logits(self, prompts: Sequence[str]) -> FloatArray:
        return np.stack([self._row for _ in prompts])


class FakeTokenizer:
    """A tokeniser backed by an explicit ``text -> ids`` table."""

    def __init__(self, table: dict[str, list[int]]) -> None:
        self._table = table

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        return list(self._table[text])


def _parse_option_lines(prompt: str) -> tuple[int, int]:
    """Count the option lines and find the one carrying the sentinel."""
    n = 0
    gold_position: int | None = None
    for line in prompt.splitlines():
        if line[:1] in _LETTER_INDEX and line[1:2] == ".":
            position = _LETTER_INDEX[line[0]]
            n += 1
            if SENTINEL in line:
                gold_position = position
    if gold_position is None:
        raise ValueError("prompt has no sentinel-marked option")
    return n, gold_position
