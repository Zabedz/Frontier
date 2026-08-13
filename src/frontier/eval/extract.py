"""Confidence extraction: cyclic scoring, PriDe debiasing, and the metrics-ready outputs.

Cyclic scoring is self-debiasing: each content visits every letter position exactly once, so
an additive per-position letter bias (the PriDe assumption) drops out of the softmax. The
prior is estimated separately only to report the bias magnitude.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from frontier.eval.prompts import build_prompt
from frontier.eval.provider import LogitProvider, letter_probs, softmax
from frontier.eval.records import (
    CorrectArray,
    EvalRecord,
    FloatArray,
    IntArray,
    LabelArray,
    ProbMatrix,
    letters_for,
)
from frontier.schema import EvalSpec, PermutationScheme, Robustness, TaskSpec

DEFAULT_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class PermutationRobustness:
    """In-flight permutation sensitivity, before the reduction to ``schema.Robustness``.

    ``letter_prior`` is averaged over the items sharing the modal option count and is carried
    for the plotting step.
    """

    letter_prior: FloatArray
    letter_bias: float
    permutation_consistency: float
    debias_flip_rate: float


@dataclass(frozen=True, slots=True)
class EvalOutputs:
    """Per-item arrays shaped to feed ``frontier.metrics``, plus the robustness record.

    ``probs`` is zero-padded to the widest option count in the scored set; ``predicted`` and
    ``gold`` always fall below an item's own count, so the padding reaches no argmax and no
    metric (a padded column is a 0.0 forecast for an event that never fires).
    """

    probs: ProbMatrix  # (n_items, max_options)
    predicted: LabelArray  # (n_items,)
    gold: LabelArray  # (n_items,)
    confidence: FloatArray  # (n_items,), == probs.max(axis=1)
    n_options: IntArray  # (n_items,), true option count per item
    robustness: PermutationRobustness | None  # None when scheme == "none"


def _cyclic_display(options: tuple[str, ...], shift: int) -> tuple[str, ...]:
    """Options under cyclic shift ``k``: content ``j`` sits at letter position ``(j + k) % n``."""
    n = len(options)
    return tuple(options[(position - shift) % n] for position in range(n))


def _score_cyclic(
    record: EvalRecord, provider: LogitProvider, candidate_ids: IntArray, *, cot: bool, eps: float
) -> tuple[FloatArray, FloatArray, float, bool]:
    """Score one item under all cyclic orders.

    Returns the debiased canonical-order distribution, the estimated per-position prior, the
    fraction of orders whose raw answer matched the debiased one, and whether the naive
    canonical-order answer differs from it.
    """
    n = len(record.options)
    prompts = [
        build_prompt(record.question, _cyclic_display(record.options, k), cot=cot) for k in range(n)
    ]
    logits = provider.next_token_logits(prompts)
    q: FloatArray = np.clip(
        np.stack([letter_probs(logits[k], candidate_ids) for k in range(n)]), eps, None
    )
    log_q = np.log(q)

    shifts = np.arange(n)
    contents = np.arange(n)
    positions = (shifts[np.newaxis, :] + contents[:, np.newaxis]) % n  # positions[j, k] = (j+k)%n
    per_content = log_q[shifts[np.newaxis, :], positions].mean(axis=1)
    debiased = softmax(per_content)
    prior = softmax(log_q.mean(axis=0))

    debiased_pick = int(np.argmax(debiased))
    raw_pick = (q.argmax(axis=1) - shifts) % n
    consistency = float(np.mean(raw_pick == debiased_pick))
    naive_flipped = int(raw_pick[0]) != debiased_pick
    return debiased, prior, consistency, naive_flipped


def _modal_option_count(counts: Sequence[int]) -> int:
    """The most common option count; ties resolve to the smaller count."""
    frequencies = np.bincount(np.asarray(counts, dtype=np.intp))
    return int(np.argmax(frequencies))


def _aggregate_robustness(
    counts: Sequence[int],
    priors: Sequence[FloatArray],
    consistencies: Sequence[float],
    flips: Sequence[bool],
) -> PermutationRobustness:
    modal = _modal_option_count(counts)
    modal_priors = [prior for prior, count in zip(priors, counts, strict=True) if count == modal]
    letter_prior: FloatArray = np.mean(np.stack(modal_priors), axis=0)
    letter_bias = float(np.max(np.abs(letter_prior - 1.0 / modal)))
    return PermutationRobustness(
        letter_prior=letter_prior,
        letter_bias=letter_bias,
        permutation_consistency=float(np.mean(consistencies)),
        debias_flip_rate=float(np.mean(flips)),
    )


def score_items(
    records: Sequence[EvalRecord],
    provider: LogitProvider,
    *,
    scheme: PermutationScheme = "cyclic",
    cot: bool = False,
    eps: float = DEFAULT_EPS,
) -> EvalOutputs:
    """Score every record and assemble the metrics-ready outputs.

    ``scheme="none"`` scores only the canonical order and leaves ``robustness`` as ``None``.
    """
    if not records:
        raise ValueError("score_items received no records")

    counts = [len(record.options) for record in records]
    n_items = len(records)
    probs = np.zeros((n_items, max(counts)), dtype=np.float64)
    predicted = np.empty(n_items, dtype=np.intp)
    gold = np.asarray([record.gold for record in records], dtype=np.intp)
    n_options = np.asarray(counts, dtype=np.intp)

    priors: list[FloatArray] = []
    consistencies: list[float] = []
    flips: list[bool] = []

    for i, record in enumerate(records):
        n = len(record.options)
        candidate_ids = provider.candidate_token_ids(letters_for(n))
        if scheme == "none":
            logits = provider.next_token_logits(
                [build_prompt(record.question, record.options, cot=cot)]
            )
            debiased = letter_probs(logits[0], candidate_ids)
        else:
            debiased, prior, consistency, flipped = _score_cyclic(
                record, provider, candidate_ids, cot=cot, eps=eps
            )
            priors.append(prior)
            consistencies.append(consistency)
            flips.append(flipped)
        probs[i, :n] = debiased
        predicted[i] = int(np.argmax(debiased))

    robustness = (
        None if scheme == "none" else _aggregate_robustness(counts, priors, consistencies, flips)
    )
    return EvalOutputs(
        probs=probs,
        predicted=predicted,
        gold=gold,
        confidence=probs.max(axis=1),
        n_options=n_options,
        robustness=robustness,
    )


def exact_match(predicted: LabelArray, gold: LabelArray) -> CorrectArray:
    """Per-item ``predicted == gold``. Accuracy is its mean."""
    result: CorrectArray = predicted == gold
    return result


def to_task_spec(spec: EvalSpec, num_items: int) -> TaskSpec:
    """Map a resolved ``schema.EvalSpec`` and the scored item count onto ``schema.TaskSpec``."""
    return TaskSpec(
        task_name=spec.task_name,
        split=spec.split,
        num_items=num_items,
        prompt_style=spec.prompt_style,
        scoring=spec.scoring,
        permutation_scheme=spec.permutation_scheme,
        labels=spec.labels,
        cot=spec.cot,
    )


def to_robustness(robustness: PermutationRobustness | None) -> Robustness | None:
    """Reduce the in-flight record onto the ``schema.Robustness`` row; ``None`` passes through.

    Only the scalars land on the row; the per-position ``letter_prior`` stays in flight.
    """
    if robustness is None:
        return None
    return Robustness(
        permutation_consistency=robustness.permutation_consistency,
        letter_bias=robustness.letter_bias,
        debias_flip_rate=robustness.debias_flip_rate,
    )
