"""The distillation losses: a hand-computed KL, the rest-bucket edge cases, and the chunked
path every arm trains through, checked against the unchunked loss and HF's own."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tiny import TINY_VOCAB, WORDS, load_tiny, tiny_llama_dir

from frontier.distill.cache import CacheBatch, teacher_targets
from frontier.distill.losses import (
    chunked_mean,
    per_position_loss,
    soft_topk_kl,
    target_tensors,
    token_nll,
)
from frontier.schema import DistillMode

torch = pytest.importorskip("torch")

# V = 5, k = 2: the student q, the teacher's two ids with their probabilities, and its rest.
HAND_Q = (0.4, 0.2, 0.2, 0.1, 0.1)
HAND_IDS = (1, 0)
HAND_PROBS = (0.5, 0.3)
HAND_REST = 0.2
HAND_KL = 0.23321130808955423  # 0.5 ln 2.5 + 0.3 ln 0.75 + 0.2 ln 0.5
HAND_CROSS_ENTROPY = 1.2628643221541278  # KL + H(P)
HAND_GRADIENT = (0.1, -0.3, 0.1, 0.05, 0.05)
FP64_TOLERANCE = 1e-12
FP32_TOLERANCE = 1e-6
STORAGE_DRIFT = 1.0004  # the fp16 cache's sum of probabilities plus rest stays within 5e-4 of 1
TINY_TEACHER_REST = 1e-30
FAR_LOGIT = -100.0
BATCH, LENGTH = 2, 9
POSITIONS = BATCH * (LENGTH - 1)
TOP_K = 64
MODES: tuple[DistillMode, ...] = ("soft_label_topk", "hard_label", "corpus_label")


def _hand(dtype: Any) -> tuple[Any, Any, Any, Any]:
    z = torch.log(torch.tensor([HAND_Q], dtype=dtype)).requires_grad_()
    return (
        z,
        torch.tensor([HAND_IDS]),
        torch.tensor([HAND_PROBS], dtype=dtype),
        torch.tensor([HAND_REST], dtype=dtype),
    )


@pytest.mark.parametrize(
    ("dtype", "tolerance"), [(torch.float64, FP64_TOLERANCE), (torch.float32, FP32_TOLERANCE)]
)
def test_the_hand_example(dtype: Any, tolerance: float) -> None:
    z, ids, probs, rest = _hand(dtype)
    loss = soft_topk_kl(z, ids, probs, rest)
    assert float(loss[0].detach()) == pytest.approx(HAND_KL, abs=tolerance)
    loss.sum().backward()
    assert z.grad[0].tolist() == pytest.approx(HAND_GRADIENT, abs=tolerance)


def test_the_hand_example_is_the_cross_entropy_less_the_target_entropy() -> None:
    z, ids, probs, rest = _hand(torch.float64)
    target = (*HAND_PROBS, HAND_REST)
    entropy = -math.fsum(p * math.log(p) for p in target)
    assert float(soft_topk_kl(z, ids, probs, rest)[0]) + entropy == pytest.approx(
        HAND_CROSS_ENTROPY, abs=FP64_TOLERANCE
    )


def test_a_teacher_whose_top_k_holds_every_bit_of_mass() -> None:
    q = (0.4, 0.3, 0.2, 0.1)
    z = torch.log(torch.tensor([q], dtype=torch.float64)).requires_grad_()
    loss = soft_topk_kl(
        z,
        torch.tensor([[0, 1]]),
        torch.tensor([[0.6, 0.4]], dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
    )
    expected = 0.6 * math.log(0.6 / 0.4) + 0.4 * math.log(0.4 / 0.3)
    assert float(loss[0].detach()) == pytest.approx(expected, abs=FP64_TOLERANCE)
    loss.sum().backward()
    assert z.grad[0].tolist() == pytest.approx([q[0] - 0.6, q[1] - 0.4, q[2], q[3]], abs=1e-12)


def test_tiny_rests_on_both_sides_stay_exact_and_keep_their_gradient() -> None:
    """A clamp on ``1 - sum`` would zero the push back into an emptied rest bucket."""
    logits = [2.0, 1.0, 0.0, FAR_LOGIT, FAR_LOGIT, FAR_LOGIT]
    z = torch.tensor([logits], dtype=torch.float32, requires_grad=True)
    ids = torch.tensor([[0, 1, 2]])
    top = (0.5, 0.3, 0.2)
    loss = soft_topk_kl(
        z, ids, torch.tensor([top]), torch.tensor([TINY_TEACHER_REST], dtype=torch.float32)
    )
    lse = math.log(math.fsum(math.exp(v) for v in logits))
    log_rest = math.log(3.0) + FAR_LOGIT - lse
    total = math.fsum(top) + TINY_TEACHER_REST
    target = [p / total for p in top]
    closed = math.fsum(
        p * (math.log(p) - (logit - lse)) for p, logit in zip(target, logits[:3], strict=True)
    )
    rest = TINY_TEACHER_REST / total
    closed += rest * (math.log(rest) - log_rest)
    assert math.isfinite(float(loss[0].detach()))
    assert float(loss[0].detach()) == pytest.approx(closed, abs=FP32_TOLERANCE)
    loss.sum().backward()
    assert torch.isfinite(z.grad).all()
    # Each off-top-k id gets -P_r q_v / Q_r = -P_r / 3, some 3e-31: compare as a ratio.
    assert (z.grad[0, 3:] / (-rest / 3)).tolist() == pytest.approx([1.0] * 3, rel=1e-3)


def test_a_rest_of_one_id_is_that_ids_probability() -> None:
    z = torch.tensor([[0.3, -1.2, 2.0, 0.7, -0.4, 1.1]], dtype=torch.float64)
    ids = torch.tensor([[2, 5, 3, 0, 4]])  # every id but 1
    probs = torch.tensor([[0.4, 0.2, 0.15, 0.1, 0.05]], dtype=torch.float64)
    rest = torch.tensor([0.1], dtype=torch.float64)
    lse = torch.logsumexp(z, dim=-1)
    log_q = torch.cat([z.gather(-1, ids) - lse, (z[:, 1] - lse)[:, None]], dim=-1)
    target = torch.cat([probs, rest[:, None]], dim=-1)
    closed = float((target * (target.log() - log_q)).sum())
    assert float(soft_topk_kl(z, ids, probs, rest)[0]) == pytest.approx(closed, abs=FP64_TOLERANCE)


def test_k_equal_to_the_vocabulary_is_refused() -> None:
    with pytest.raises(ValueError, match="no rest bucket"):
        soft_topk_kl(
            torch.zeros(1, 3),
            torch.tensor([[0, 1, 2]]),
            torch.full((1, 3), 1 / 3),
            torch.zeros(1),
        )


def test_storage_drift_in_the_target_is_normalised_away() -> None:
    z, ids, probs, rest = _hand(torch.float32)
    drifted = soft_topk_kl(z, ids, probs * STORAGE_DRIFT, rest * STORAGE_DRIFT)
    assert float(drifted[0]) == pytest.approx(float(soft_topk_kl(z, ids, probs, rest)[0]), abs=1e-7)


def test_a_student_equal_to_its_teacher_has_zero_kl() -> None:
    generator = torch.Generator().manual_seed(0)
    z = 2.0 * torch.randn(8, 50, generator=generator, dtype=torch.float64)
    ids = z.topk(10, dim=-1).indices
    rest = torch.exp(torch.logsumexp(z.scatter(-1, ids, float("-inf")), -1) - z.logsumexp(-1))
    exact = soft_topk_kl(z, ids, z.softmax(-1).gather(-1, ids), rest)
    assert exact.abs().max().item() == pytest.approx(0.0, abs=1e-7)


def test_fp16_storage_of_the_target_leaves_a_negligible_kl() -> None:
    generator = torch.Generator().manual_seed(0)
    z = 2.0 * torch.randn(8, 50, generator=generator)
    ids, probs, rest = teacher_targets(z, 10)
    # Rounding to fp16 moves each probability by at most 2^-11 relative: a KL of order 1e-7.
    assert soft_topk_kl(z, ids, probs.float(), rest).abs().max().item() == pytest.approx(
        0.0, abs=FP32_TOLERANCE
    )


def test_token_nll_is_the_negative_log_softmax_of_the_target() -> None:
    generator = torch.Generator().manual_seed(1)
    z = torch.randn(6, 11, generator=generator)
    targets = torch.tensor([0, 3, 10, 5, 5, 2])
    expected = -z.log_softmax(-1).gather(-1, targets[:, None])[:, 0]
    assert torch.allclose(token_nll(z, targets), expected, atol=FP32_TOLERANCE)
    assert torch.allclose(
        token_nll(z, targets).mean(),
        torch.nn.functional.cross_entropy(z, targets),
        atol=FP32_TOLERANCE,
    )


def _batch(model: Any) -> CacheBatch:
    """Random word ids, with a second tiny model's targets as the teacher."""
    generator = torch.Generator().manual_seed(2)
    ids = torch.randint(2, WORDS + 2, (BATCH, LENGTH), generator=generator)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[:, :-1]
    topk, probs, rest = teacher_targets(logits, TOP_K)
    return CacheBatch(
        input_ids=ids.numpy().astype(np.int64),
        topk_ids=topk.numpy().astype(np.int64),
        topk_probs=probs.float().numpy(),
        rest=rest.numpy(),
    )


def _loss_and_grads(
    model: Any, batch: CacheBatch, mode: DistillMode, chunk: int | None
) -> tuple[float, list[Any]]:
    model.zero_grad(set_to_none=True)
    hidden = model.get_decoder()(input_ids=torch.from_numpy(batch.input_ids)).last_hidden_state
    hidden = hidden[:, :-1]
    head = model.get_output_embeddings()
    targets = target_tensors(batch, mode, device="cpu")
    per_position = per_position_loss(mode)
    if chunk is None:
        flat = [t.reshape(POSITIONS, *t.shape[2:]) for t in targets]
        loss = per_position(head(hidden.reshape(POSITIONS, -1)).float(), *flat).mean()
    else:
        loss = chunked_mean(hidden, head, per_position, *targets, chunk=chunk)
    loss.backward()
    return float(loss), [p.grad.clone() for p in model.parameters()]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("chunk", [1, 7, POSITIONS, POSITIONS + 5])
def test_the_chunked_mean_equals_the_unchunked_loss_and_gradients(
    tmp_path: Path, mode: DistillMode, chunk: int
) -> None:
    student = load_tiny(tiny_llama_dir(tmp_path, seed=0))
    batch = _batch(load_tiny(tiny_llama_dir(tmp_path, name="teacher", seed=1)))
    whole, whole_grads = _loss_and_grads(student, batch, mode, None)
    chunked, chunked_grads = _loss_and_grads(student, batch, mode, chunk)
    assert chunked == pytest.approx(whole, abs=FP32_TOLERANCE)
    assert len(chunked_grads) == len(whole_grads)
    for got, want in zip(chunked_grads, whole_grads, strict=True):
        assert torch.allclose(got, want, atol=FP32_TOLERANCE)


def test_the_corpus_arm_reproduces_the_hf_next_token_loss(tmp_path: Path) -> None:
    model = load_tiny(tiny_llama_dir(tmp_path))
    batch = _batch(model)
    chunked, chunked_grads = _loss_and_grads(model, batch, "corpus_label", 5)
    model.zero_grad(set_to_none=True)
    ids = torch.from_numpy(batch.input_ids)
    reference = model(input_ids=ids, labels=ids).loss
    reference.backward()
    assert chunked == pytest.approx(float(reference), abs=FP32_TOLERANCE)
    for got, parameter in zip(chunked_grads, model.parameters(), strict=True):
        assert torch.allclose(got, parameter.grad, atol=FP32_TOLERANCE)


def test_the_split_forward_is_the_models_own_logits(tmp_path: Path) -> None:
    model = load_tiny(tiny_llama_dir(tmp_path))
    ids = torch.randint(0, TINY_VOCAB, (BATCH, LENGTH), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        split = model.get_output_embeddings()(model.get_decoder()(input_ids=ids).last_hidden_state)
        assert torch.allclose(split, model(input_ids=ids).logits, atol=FP32_TOLERANCE)


def test_target_tensors_carry_what_each_mode_reads(tmp_path: Path) -> None:
    batch = _batch(load_tiny(tiny_llama_dir(tmp_path)))
    assert batch.topk_ids is not None
    soft = target_tensors(batch, "soft_label_topk", device="cpu")
    assert [tuple(t.shape) for t in soft] == [
        (BATCH, LENGTH - 1, TOP_K),
        (BATCH, LENGTH - 1, TOP_K),
        (BATCH, LENGTH - 1),
    ]
    assert [t.dtype for t in soft] == [torch.int64, torch.float32, torch.float32]
    (hard,) = target_tensors(batch, "hard_label", device="cpu")
    assert torch.equal(hard, torch.from_numpy(batch.topk_ids[..., 0]))
    (corpus,) = target_tensors(batch, "corpus_label", device="cpu")
    assert torch.equal(corpus, torch.from_numpy(batch.input_ids[:, 1:]))


def test_a_soft_batch_without_teacher_arrays_is_refused() -> None:
    bare = CacheBatch(np.zeros((1, 4), dtype=np.int64), None, None, None)
    with pytest.raises(ValueError, match="cached teacher ids"):
        target_tensors(bare, "hard_label", device="cpu")
