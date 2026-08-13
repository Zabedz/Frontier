"""The ``LogitProvider`` protocol and the letter-token primitives. The eval core reaches a
model only through this seam, so it imports no inference stack.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from frontier.eval.records import FloatArray, IntArray

SINGLE_TOKEN = 1


@runtime_checkable
class Tokenizer(Protocol):
    """The one tokeniser call candidate-id resolution needs."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@runtime_checkable
class LogitProvider(Protocol):
    """Next-token logits for a batch of prompts, plus the answer-letter token ids."""

    def candidate_token_ids(self, letters: Sequence[str]) -> IntArray:
        """One token id per answer letter, in the same order as ``letters``."""
        ...

    def next_token_logits(self, prompts: Sequence[str]) -> FloatArray:
        """Next-token logits for a batch of prompts, shape ``(len(prompts), vocab)``."""
        ...


def resolve_candidate_ids(
    tokenizer: Tokenizer, letters: Sequence[str], *, leading_space: bool = True
) -> IntArray:
    """Map each answer letter to its single next-token id.

    ``leading_space`` encodes ``" " + letter``, the token that follows a prompt ending in
    ``"Answer:"``; the bare form suits a trigger that already ends in a space. A letter that
    does not encode to exactly one id raises ``ValueError``.
    """
    ids: list[int] = []
    for letter in letters:
        text = f" {letter}" if leading_space else letter
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != SINGLE_TOKEN:
            raise ValueError(
                f"answer letter {letter!r} encoded as {text!r} is not a single token: got {encoded}"
            )
        ids.append(encoded[0])
    return np.asarray(ids, dtype=np.intp)


def softmax(x: FloatArray) -> FloatArray:
    """Numerically stable softmax over a 1-D vector (subtract the max)."""
    exp = np.exp(x - np.max(x))
    result: FloatArray = exp / np.sum(exp)
    return result


def letter_probs(logits: FloatArray, candidate_ids: IntArray) -> FloatArray:
    """Softmax over only the candidate-letter logits.

    Scoring this single-token signal avoids the length-normalisation confound a
    full-option-string score carries.
    """
    return softmax(logits[candidate_ids])
