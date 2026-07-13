"""The logit-provider protocol and the letter-token primitives.

The eval core never imports a model. It obtains next-token logits through the narrow
``LogitProvider`` protocol, which the real Hugging Face backend (next WP) and the
synthetic test provider both satisfy. Candidate-id resolution and the two numeric
primitives are pure functions the backend and the tests share.
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
        """One token id per answer letter, in the same order as ``letters``.

        The single id read at the answer position for each letter; the letter set is
        fixed across an item's permutations, so this is resolved once per item.
        """
        ...

    def next_token_logits(self, prompts: Sequence[str]) -> FloatArray:
        """Next-token logits for a batch of prompts, shape ``(len(prompts), vocab)``."""
        ...


def resolve_candidate_ids(
    tokenizer: Tokenizer, letters: Sequence[str], *, leading_space: bool = True
) -> IntArray:
    """Map each answer letter to its single next-token id.

    With ``leading_space=True`` (the default, matching a prompt that ends in
    ``"Answer:"`` with no trailing space) the letter is encoded as ``" " + letter``,
    the token that follows the colon. With ``leading_space=False`` the bare letter is
    encoded, the convention when the trigger already ends in a space or the letter
    opens a fresh assistant turn.

    Each candidate must tokenise to exactly one id; otherwise raise ``ValueError``
    naming the letter, the encoded string, and the id list, so a tokeniser that
    splits a letter fails loudly rather than silently reading the wrong logit.
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

    Gathers ``logits[candidate_ids]`` and softmaxes them, giving a per-option
    distribution over exactly the answer letters. This is the single-token signal
    from ``docs/decisions.md``: no full-option-string scoring, so no
    length-normalisation confound.
    """
    return softmax(logits[candidate_ids])
