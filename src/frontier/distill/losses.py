"""The per-position distillation losses (a top-k-plus-rest KL, token cross-entropy) and the
chunked mean every arm trains through."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from frontier.schema import DistillMode

if TYPE_CHECKING:
    from frontier.distill.cache import CacheBatch

LOSS_CHUNK_POSITIONS = 1024
PerPosition = Callable[..., Any]  # (logits (n, V) fp32, *targets (n, ...)) -> (n,) losses


def soft_topk_kl(logits: Any, ids: Any, probs: Any, rest: Any) -> Any:
    """Per-position KL from the teacher's top-k-plus-rest target to the student.

    The target ``(probs, rest)`` is divided by its own sum, which removes the fp16 storage
    rounding and keeps the rest bucket its mass. Raises ``ValueError`` when k >= V.
    """
    import torch  # noqa: PLC0415

    vocab = int(logits.shape[-1])
    if ids.shape[-1] >= vocab:
        raise ValueError(f"k = {ids.shape[-1]} leaves no rest bucket in a {vocab}-entry vocabulary")
    lse = torch.logsumexp(logits, dim=-1)
    log_top = logits.gather(-1, ids) - lse[..., None]
    log_rest = torch.logsumexp(logits.scatter(-1, ids, float("-inf")), dim=-1) - lse
    target = torch.cat([probs, rest[..., None]], dim=-1)
    target = target / target.sum(dim=-1, keepdim=True)
    log_student = torch.cat([log_top, log_rest[..., None]], dim=-1)
    return (torch.special.xlogy(target, target) - target * log_student).sum(dim=-1)


def token_nll(logits: Any, targets: Any) -> Any:
    """Per-position cross-entropy, ``torch.nn.functional.cross_entropy(reduction="none")``."""
    import torch  # noqa: PLC0415

    return torch.nn.functional.cross_entropy(logits, targets, reduction="none")


def chunked_mean(
    hidden: Any,
    head: Any,
    per_position: PerPosition,
    *targets: Any,
    chunk: int = LOSS_CHUNK_POSITIONS,
) -> Any:
    """Mean of ``per_position(head(h).float(), *t)`` over every position.

    One chunk of positions at a time under ``torch.utils.checkpoint`` (``use_reentrant=False``),
    so a chunk's full-vocabulary logits are freed after its forward and rebuilt for its
    backward, in the autocast state the forward ran under.
    """
    from torch.utils.checkpoint import checkpoint  # noqa: PLC0415

    flat = hidden.reshape(-1, hidden.shape[-1])
    n_positions = int(flat.shape[0])
    flat_targets = [target.reshape(n_positions, *target.shape[2:]) for target in targets]
    total: Any = None
    for start in range(0, n_positions, chunk):
        part = checkpoint(
            partial(_chunk_sum, head, per_position),
            flat[start : start + chunk],
            *(target[start : start + chunk] for target in flat_targets),
            use_reentrant=False,
        )
        total = part if total is None else total + part
    return total / n_positions


def per_position_loss(mode: DistillMode) -> PerPosition:
    """The per-position loss for ``mode``; its targets come from ``target_tensors``."""
    return soft_topk_kl if mode == "soft_label_topk" else token_nll


def target_tensors(batch: CacheBatch, mode: DistillMode, *, device: str) -> tuple[Any, ...]:
    """soft: (ids int64, probs float32, rest float32); hard: (topk_ids[..., 0],);
    corpus: (input_ids[:, 1:],). Each shaped (B, L - 1, ...)."""
    import torch  # noqa: PLC0415

    if mode == "corpus_label":
        arrays: tuple[Any, ...] = (batch.input_ids[:, 1:],)
    elif batch.topk_ids is None:
        raise ValueError(f"a {mode} batch needs the cached teacher ids")
    elif mode == "hard_label":
        arrays = (batch.topk_ids[..., 0],)
    elif batch.topk_probs is None or batch.rest is None:
        raise ValueError("a soft_label_topk batch needs the cached probabilities and rest")
    else:
        arrays = (batch.topk_ids, batch.topk_probs, batch.rest)
    return tuple(torch.from_numpy(np.ascontiguousarray(array)).to(device) for array in arrays)


def _chunk_sum(head: Any, per_position: PerPosition, hidden: Any, *targets: Any) -> Any:
    return per_position(head(hidden).float(), *targets).sum()
