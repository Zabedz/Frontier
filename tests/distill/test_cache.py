"""The teacher cache: target maths, layout, alignment, resume, reuse, and every refusal."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from tiny import (
    CONFIG_ROOT,
    ONE_STEP_TOKENS,
    SHORT_ROW,
    arithmetic_rows,
    build_cache,
    cpu_facts,
    load_tiny,
    request_for,
    successor_forward,
    tiny_llama_dir,
    tiny_variant,
    word_texts,
)

from frontier.distill import build
from frontier.distill.build import cache_request, ensure_cache
from frontier.distill.cache import (
    INPUT_IDS_FILE,
    META_FILE,
    REST_FILE,
    TEACHER_BATCH_ROWS,
    TOPK_IDS_FILE,
    TOPK_PROBS_FILE,
    CacheMismatchError,
    TeacherCache,
    fill_targets,
    read_meta,
    start_cache,
    teacher_targets,
)
from frontier.distill.fidelity import CacheFidelityError
from frontier.distill.heldout import BASE_HELDOUT_FILE, read_heldout
from frontier.distill.paths import cache_dir
from frontier.pipeline.config import resolve_config

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ARRAYS = (INPUT_IDS_FILE, TOPK_IDS_FILE, TOPK_PROBS_FILE, REST_FILE)
SMOKE_TRAIN_ROWS = ONE_STEP_TOKENS // 1024
SMOKE_HELDOUT = 4
FULL_TRAIN_ROWS = 29_296
FULL_HELDOUT = 64
CRASH_ON_CALL = 3
ROWS_BEFORE_CRASH = 4  # two flushed two-row batches
CONFIDENT_LOGIT = 30.0  # four other ids hold 4 e^-30 = 3.7e-13 of the mass
DRIFT = 1.5  # a temperature, since a constant shift would leave every probability alone


def _texts(_corpus: str) -> list[str]:
    return word_texts()


def _no_texts(_corpus: str) -> list[str]:
    raise AssertionError("packed the corpus before refusing")


def _ensure(variant: Any, root: Path, *, max_tokens: int = ONE_STEP_TOKENS) -> TeacherCache:
    return ensure_cache(
        variant, checkpoints_root=root, device="cpu", max_tokens=max_tokens, texts=_texts
    )


def _refuse_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a complete cache loaded a model")

    for auto in (transformers.AutoModelForCausalLM, transformers.AutoTokenizer):
        monkeypatch.setattr(auto, "from_pretrained", refuse)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", refuse)


def test_teacher_targets_on_a_hand_example() -> None:
    ids, probs, rest = teacher_targets(torch.tensor([3.0, 1.0, 2.0, 0.0, -1.0]), 2)
    assert ids.tolist() == [0, 2]
    assert probs.tolist() == [0.63623046875, 0.234130859375]
    assert float(rest) == pytest.approx(0.12946969618843257, rel=1e-6)
    assert (ids.dtype, probs.dtype, rest.dtype) == (torch.int64, torch.float16, torch.float32)


def test_a_confident_teachers_rest_survives_where_one_minus_the_sum_is_zero() -> None:
    _ids, probs, rest = teacher_targets(torch.tensor([CONFIDENT_LOGIT, 0.0, 0.0, 0.0, 0.0]), 1)
    assert float(1.0 - probs.float().sum()) == 0.0
    assert float(rest) == pytest.approx(4.0 * math.exp(-CONFIDENT_LOGIT), rel=1e-5, abs=0.0)


def test_teacher_targets_refuse_a_non_finite_logit_and_name_the_row() -> None:
    logits = torch.zeros(3, 6)
    logits[1, 4] = float("inf")
    with pytest.raises(FloatingPointError, match=r"row 17 index \(1, 4\)"):
        teacher_targets(logits, 2, row=17)


def test_teacher_targets_need_a_rest_bucket() -> None:
    with pytest.raises(ValueError, match="no rest bucket"):
        teacher_targets(torch.zeros(2, 4), 4)


def test_the_arrays_on_disk_have_the_documented_shapes_and_dtypes(tmp_path: Path) -> None:
    rows = arithmetic_rows(6)
    meta = build_cache(tmp_path, rows, successor_forward(), top_k=3)
    expected = {
        INPUT_IDS_FILE: ((6, SHORT_ROW), np.uint32),
        TOPK_IDS_FILE: ((6, SHORT_ROW - 1, 3), np.uint32),
        TOPK_PROBS_FILE: ((6, SHORT_ROW - 1, 3), np.float16),
        REST_FILE: ((6, SHORT_ROW - 1), np.float32),
    }
    for name, (shape, dtype) in expected.items():
        stored = np.load(tmp_path / name, mmap_mode="r")
        assert (stored.shape, stored.dtype) == (shape, dtype), name
    assert meta.complete
    assert meta.rest_summary is not None
    assert set(meta.rest_summary) == {"mean", "p50", "p99", "max"}


def test_position_t_holds_the_teacher_distribution_over_token_t_plus_one(tmp_path: Path) -> None:
    rows = arithmetic_rows(6)
    build_cache(tmp_path, rows, successor_forward(), top_k=3)
    topk_ids = np.load(tmp_path / TOPK_IDS_FILE)
    assert np.array_equal(topk_ids[:, :, 0], rows[:, 1:])


def test_column_zero_is_the_teachers_argmax(tmp_path: Path) -> None:
    model = load_tiny(tiny_llama_dir(tmp_path))
    rows = arithmetic_rows(4)

    def forward(ids: Any) -> Any:
        return model(input_ids=ids).logits

    build_cache(tmp_path / "cache", rows, forward, top_k=5)
    with torch.no_grad():
        argmax = model(input_ids=torch.from_numpy(rows)).logits[:, :-1].argmax(-1)
    assert np.array_equal(np.load(tmp_path / "cache" / TOPK_IDS_FILE)[:, :, 0], argmax.numpy())


def test_a_resumed_build_equals_an_uninterrupted_one_bit_for_bit(tmp_path: Path) -> None:
    rows = arithmetic_rows(6)
    forward = successor_forward()
    calls = 0

    def crashing(ids: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == CRASH_ON_CALL:
            raise RuntimeError("simulated crash")
        return forward(ids)

    facts = cpu_facts()
    resumed = tmp_path / "resumed"
    start_cache(resumed, rows, request_for(rows, top_k=3), facts, batch_rows=2, flush_rows=2)
    with pytest.raises(RuntimeError, match="simulated crash"):
        fill_targets(resumed, crashing, facts=facts, device="cpu")
    assert read_meta(resumed).rows_done == ROWS_BEFORE_CRASH
    fill_targets(resumed, forward, facts=facts, device="cpu")
    build_cache(tmp_path / "whole", rows, forward, top_k=3)
    for name in ARRAYS:
        assert np.array_equal(
            np.load(resumed / name), np.load(tmp_path / "whole" / name), equal_nan=True
        ), name


def test_resuming_on_another_device_is_refused(tmp_path: Path) -> None:
    rows = arithmetic_rows(4)
    start_cache(tmp_path, rows, request_for(rows, top_k=3), cpu_facts())
    stored = json.loads((tmp_path / META_FILE).read_text(encoding="utf-8"))
    stored["device"] = "NVIDIA RTX 4000 Ada Generation"
    (tmp_path / META_FILE).write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(CacheMismatchError, match="device"):
        fill_targets(tmp_path, successor_forward(), facts=cpu_facts(), device="cpu")


def test_an_edited_row_breaks_the_fingerprint(tmp_path: Path) -> None:
    build_cache(tmp_path, arithmetic_rows(4), successor_forward(), top_k=3)
    TeacherCache(tmp_path)
    stored = np.load(tmp_path / INPUT_IDS_FILE, mmap_mode="r+")
    stored[2, 3] += 1
    stored.flush()
    del stored
    with pytest.raises(ValueError, match="rows fingerprint"):
        TeacherCache(tmp_path)


def test_an_incomplete_cache_is_not_served(tmp_path: Path) -> None:
    rows = arithmetic_rows(4)
    start_cache(tmp_path, rows, request_for(rows, top_k=3), cpu_facts())
    with pytest.raises(ValueError, match="incomplete"):
        TeacherCache(tmp_path)


def test_a_row_outside_the_teacher_vocabulary_is_refused(tmp_path: Path) -> None:
    rows = arithmetic_rows(2)
    rows[0, 0] = cpu_facts().vocab_size
    with pytest.raises(ValueError, match="outside the teacher"):
        start_cache(tmp_path, rows, request_for(rows, top_k=3), cpu_facts())


def test_the_shipped_request_holds_out_rows_past_the_training_rows() -> None:
    variant = resolve_config(
        CONFIG_ROOT / "variants" / "distill-softlabel-topk.yaml", config_root=CONFIG_ROOT
    ).variant
    full = cache_request(variant)
    assert (full.train_rows, full.heldout_rows, full.capped) == (
        FULL_TRAIN_ROWS,
        FULL_HELDOUT,
        False,
    )
    capped = cache_request(variant, max_tokens=ONE_STEP_TOKENS)
    assert (capped.train_rows, capped.heldout_rows, capped.capped) == (
        SMOKE_TRAIN_ROWS,
        SMOKE_HELDOUT,
        True,
    )


def test_ensure_cache_builds_a_checked_cache_with_disjoint_held_out_rows(tmp_path: Path) -> None:
    variant = tiny_variant(tiny_llama_dir(tmp_path))
    cache = _ensure(variant, tmp_path / "ckpt")
    assert cache.path == cache_dir(variant, root=tmp_path / "ckpt")
    assert (len(cache.training), len(cache.heldout)) == (SMOKE_TRAIN_ROWS, SMOKE_HELDOUT)
    assert not set(cache.training) & set(cache.heldout)
    assert cache.meta.capped
    assert cache.meta.deterministic
    assert not torch.are_deterministic_algorithms_enabled()
    base = read_heldout(cache.path / BASE_HELDOUT_FILE)
    assert (base.rows, base.positions) == (SMOKE_HELDOUT, SMOKE_HELDOUT * 1023)
    assert np.isfinite([base.soft_kl, base.hard_nll, base.corpus_nll]).all()


def test_a_complete_cache_returns_without_loading_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variant = tiny_variant(tiny_llama_dir(tmp_path))
    built = _ensure(variant, tmp_path / "ckpt")
    _refuse_loads(monkeypatch)
    assert _ensure(variant, tmp_path / "ckpt").meta == built.meta


def test_a_larger_complete_cache_serves_a_smaller_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variant = tiny_variant(tiny_llama_dir(tmp_path))
    _ensure(variant, tmp_path / "ckpt")
    _refuse_loads(monkeypatch)
    smaller = _ensure(variant, tmp_path / "ckpt", max_tokens=ONE_STEP_TOKENS // 2)
    assert len(smaller.training) == SMOKE_TRAIN_ROWS


def test_a_cache_with_fewer_rows_than_the_request_is_refused(tmp_path: Path) -> None:
    variant = tiny_variant(tiny_llama_dir(tmp_path))
    _ensure(variant, tmp_path / "ckpt", max_tokens=ONE_STEP_TOKENS // 4)
    with pytest.raises(CacheMismatchError, match="holds 4 training rows"):
        _ensure(variant, tmp_path / "ckpt")


def test_a_cache_built_for_another_k_or_revision_is_refused(tmp_path: Path) -> None:
    variant = tiny_variant(tiny_llama_dir(tmp_path))
    cache = _ensure(variant, tmp_path / "ckpt")
    pinned = replace(variant, model=replace(variant.model, model_revision="v2"))
    with pytest.raises(CacheMismatchError, match="teacher_revision='main'"):
        _ensure(pinned, tmp_path / "ckpt")
    stored = json.loads((cache.path / META_FILE).read_text(encoding="utf-8"))
    stored["top_k"] = 32
    (cache.path / META_FILE).write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(CacheMismatchError, match="top_k=32"):
        _ensure(variant, tmp_path / "ckpt")


def test_a_student_with_another_tokenizer_is_refused_before_packing(tmp_path: Path) -> None:
    teacher = tiny_llama_dir(tmp_path)
    student = tiny_llama_dir(tmp_path, name="other-words", words=50)
    variant = tiny_variant(teacher, student_dir=student)
    with pytest.raises(ValueError, match="one tokenizer"):
        ensure_cache(
            variant,
            checkpoints_root=tmp_path / "ckpt",
            device="cpu",
            max_tokens=ONE_STEP_TOKENS,
            texts=_no_texts,
        )


def test_k_at_the_vocabulary_size_is_refused(tmp_path: Path) -> None:
    variant = tiny_variant(tiny_llama_dir(tmp_path, vocab=64))
    with pytest.raises(ValueError, match="no rest bucket"):
        ensure_cache(
            variant,
            checkpoints_root=tmp_path / "ckpt",
            device="cpu",
            max_tokens=ONE_STEP_TOKENS,
            texts=_no_texts,
        )


class _Drifting:
    """The tiny Llama on its first call, with its logits rescaled on every later call."""

    def __init__(self, model: Any) -> None:
        self.model, self.config, self.calls = model, model.config, 0

    def to(self, _device: str) -> _Drifting:
        return self

    def eval(self) -> _Drifting:
        return self

    def __call__(self, *, input_ids: Any) -> Any:
        self.calls += 1
        logits = self.model(input_ids=input_ids).logits
        return SimpleNamespace(logits=logits if self.calls == 1 else logits * DRIFT)


def test_a_fresh_build_checks_its_first_batch_after_the_first_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tiny_llama_dir(tmp_path)
    variant = tiny_variant(model_dir)
    teacher = _Drifting(load_tiny(model_dir))
    monkeypatch.setattr(build, "FLUSH_EVERY_ROWS", TEACHER_BATCH_ROWS)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *_a, **_k: teacher
    )
    with pytest.raises(CacheFidelityError, match="row 0 position 0"):
        _ensure(variant, tmp_path / "ckpt")
    meta = read_meta(cache_dir(variant, root=tmp_path / "ckpt"))
    assert meta.rows_done == TEACHER_BATCH_ROWS
    assert not meta.complete


def test_a_relaunch_after_a_failed_first_check_repeats_it_before_filling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tiny_llama_dir(tmp_path)
    variant = tiny_variant(model_dir)
    teacher = _Drifting(load_tiny(model_dir))
    monkeypatch.setattr(build, "FLUSH_EVERY_ROWS", TEACHER_BATCH_ROWS)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *_a, **_k: teacher
    )
    for _attempt in range(2):
        with pytest.raises(CacheFidelityError, match="row 0 position 0"):
            _ensure(variant, tmp_path / "ckpt")
        assert read_meta(cache_dir(variant, root=tmp_path / "ckpt")).rows_done == (
            TEACHER_BATCH_ROWS
        )


def test_the_base_report_belongs_to_one_student(tmp_path: Path) -> None:
    variant = tiny_variant(tiny_llama_dir(tmp_path))
    cache = _ensure(variant, tmp_path / "ckpt")
    path = cache.path / BASE_HELDOUT_FILE
    assert read_heldout(path).student_id == variant.model.student_id
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["student_id"] = "another/student"
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValueError, match="scores the untrained another/student"):
        _ensure(variant, tmp_path / "ckpt")
