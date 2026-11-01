"""The distillation trainer, offline on the tiny Llama: reuse, refusals, and one real step
per arm from one shared cache."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tiny import ARMS, ONE_STEP_TOKENS, tiny_llama_dir, tiny_variant, word_texts

from frontier.distill import build
from frontier.distill.cache import TeacherCache
from frontier.distill.heldout import HELDOUT_FILE, read_heldout
from frontier.distill.losses import LOSS_CHUNK_POSITIONS
from frontier.distill.paths import STUDENT_METADATA_FILE, student_path
from frontier.distill.train import produce_distill
from frontier.schema import VariantConfig

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

SMOKE_HELDOUT = 4
ARM_MODES = {
    "distill-softlabel-topk": "soft_label_topk",
    "distill-corpus": "corpus_label",
    "distill-hardlabel": "hard_label",
}
SEQ_LEN = 1024
SOFT_LOSS_AT_IDENTITY = 1e-6


def _texts(_corpus: str) -> list[str]:
    return word_texts()


def _finished(variant: VariantConfig, root: Path, *, budget: int = ONE_STEP_TOKENS) -> Path:
    """A student directory whose markers record ``variant``'s recipe at ``budget``."""
    assert variant.distill is not None
    out = student_path(variant, root=root)
    out.mkdir(parents=True)
    (out / "config.json").write_text("{}", encoding="utf-8")
    metadata = {
        "recipe": {"model": asdict(variant.model), "distill": asdict(variant.distill)},
        "budget_tokens": budget,
    }
    (out / STUDENT_METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")
    return out


def _refuse_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a finished student loaded a model")

    for auto in (transformers.AutoModelForCausalLM, transformers.AutoTokenizer):
        monkeypatch.setattr(auto, "from_pretrained", refuse)


def _produce(variant: VariantConfig, root: Path) -> Path:
    return produce_distill(
        variant,
        checkpoints_root=root,
        seed=0,
        device="cpu",
        max_tokens=ONE_STEP_TOKENS,
        texts=_texts,
    )


def test_a_finished_student_with_the_same_recipe_is_returned_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variant = tiny_variant(tmp_path / "unused")
    out = _finished(variant, tmp_path)
    _refuse_loads(monkeypatch)
    assert _produce(variant, tmp_path) == out


def test_a_finished_student_with_another_learning_rate_is_refused(tmp_path: Path) -> None:
    variant = tiny_variant(tmp_path / "unused")
    assert variant.distill is not None
    _finished(variant, tmp_path)
    faster = replace(variant, distill=replace(variant.distill, learning_rate=2.0e-5))
    with pytest.raises(ValueError, match=r"distill\.learning_rate=1e-05"):
        _produce(faster, tmp_path)


def test_a_finished_student_on_another_budget_is_refused(tmp_path: Path) -> None:
    variant = tiny_variant(tmp_path / "unused")
    _finished(variant, tmp_path, budget=ONE_STEP_TOKENS // 2)
    with pytest.raises(ValueError, match="trained on 8192 tokens"):
        _produce(variant, tmp_path)


def _counting(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    run_teacher: Callable[..., None] = build._run_teacher

    def counted(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        run_teacher(*args, **kwargs)

    monkeypatch.setattr(build, "_run_teacher", counted)
    return calls


def test_the_three_arms_train_one_step_each_from_one_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tiny_llama_dir(tmp_path)
    teacher_runs = _counting(monkeypatch)
    root = tmp_path / "ckpt"
    stored: dict[str, Any] = {}
    for name in ARMS:
        out = _produce(tiny_variant(model_dir, name=name), root)
        stored[name] = json.loads((out / STUDENT_METADATA_FILE).read_text(encoding="utf-8"))
        assert (out / "config.json").is_file()
        heldout = read_heldout(out / HELDOUT_FILE)
        assert (heldout.rows, heldout.positions) == (SMOKE_HELDOUT, SMOKE_HELDOUT * (SEQ_LEN - 1))
        assert np.isfinite([heldout.soft_kl, heldout.hard_nll, heldout.corpus_nll]).all()
        weights = safetensors_torch.load_file(str(out / "model.safetensors"))
        assert {tensor.dtype for tensor in weights.values()} == {torch.float16}
        transformers.AutoModelForCausalLM.from_pretrained(out)
    assert len(teacher_runs) == 1
    for name, metadata in stored.items():
        assert metadata["mode"] == ARM_MODES[name]
        assert (metadata["steps"], metadata["tokens_trained"]) == (1, ONE_STEP_TOKENS)
        assert metadata["capped"]
        assert metadata["saved_dtype"] == "float16"
        assert np.isfinite(metadata["final_loss"])
    assert len({metadata["cache"]["rows_fingerprint"] for metadata in stored.values()}) == 1


def test_a_student_identical_to_its_teacher_starts_at_zero_soft_loss(tmp_path: Path) -> None:
    """An off-by-one between the cache and the student forward would show as a large KL."""
    out = _produce(tiny_variant(tiny_llama_dir(tmp_path)), tmp_path / "ckpt")
    metadata = json.loads((out / STUDENT_METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["steps"] == 1
    assert abs(metadata["final_loss"]) < SOFT_LOSS_AT_IDENTITY


def test_the_trainer_reads_training_rows_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read: list[int] = []
    batch = TeacherCache.batch

    def recorded(cache: TeacherCache, rows: Any, mode: Any) -> Any:
        if rows.min() < len(cache.training):
            read.extend(int(row) for row in rows)
        return batch(cache, rows, mode)

    monkeypatch.setattr(TeacherCache, "batch", recorded)
    out = _produce(tiny_variant(tiny_llama_dir(tmp_path), name="distill-corpus"), tmp_path / "ckpt")
    assert sorted(read) == list(range(ONE_STEP_TOKENS // SEQ_LEN))
    metadata = json.loads((out / STUDENT_METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["loss_chunk_positions"] == LOSS_CHUNK_POSITIONS
    assert transformers.AutoConfig.from_pretrained(out).use_cache


def test_a_budget_under_one_step_fails_before_the_teacher_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher_runs = _counting(monkeypatch)
    variant = tiny_variant(tiny_llama_dir(tmp_path))
    with pytest.raises(ValueError, match="cannot fill one 16-row"):
        produce_distill(
            variant,
            checkpoints_root=tmp_path / "ckpt",
            seed=0,
            device="cpu",
            max_tokens=ONE_STEP_TOKENS // 4,
            texts=_texts,
        )
    assert teacher_runs == []
