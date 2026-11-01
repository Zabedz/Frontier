"""Distill config resolution: the shipped arms, the scored model, and the resolve-time refusals."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from frontier.distill.spec import require_distill_spec, scored_model, teacher_model
from frontier.pipeline.config import ResolvedConfig, resolve_config
from frontier.schema import RunMode

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
SOFT = CONFIG_ROOT / "variants" / "distill-softlabel-topk.yaml"
QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"
QWEN_05B = "Qwen/Qwen2.5-0.5B-Instruct"
SMOLLM2 = "HuggingFaceTB/SmolLM2-135M-Instruct"
ARM_MODES = {
    "distill-softlabel-topk": "soft_label_topk",
    "distill-hardlabel": "hard_label",
    "distill-corpus": "corpus_label",
}


def _resolve(name: str, mode: RunMode = "full") -> ResolvedConfig:
    return resolve_config(
        CONFIG_ROOT / "variants" / f"{name}.yaml", mode=mode, config_root=CONFIG_ROOT
    )


def _variant_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "variant.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ["full", "smoke"])
@pytest.mark.parametrize(("name", "expected"), sorted(ARM_MODES.items()))
def test_each_shipped_arm_resolves_to_its_mode(name: str, expected: str, mode: RunMode) -> None:
    spec = require_distill_spec(_resolve(name, mode).variant)
    assert spec.mode == expected
    assert (spec.corpus, spec.top_k, spec.train_tokens, spec.seq_len) == (
        "fineweb_edu",
        64,
        30_000_000,
        1024,
    )
    assert spec.learning_rate == pytest.approx(1.0e-5)


@pytest.mark.parametrize(
    ("name", "mode", "expected"),
    [
        ("distill-softlabel-topk", "full", QWEN_05B),
        ("distill-softlabel-topk", "smoke", SMOLLM2),
        ("fp16", "full", QWEN_3B),
        ("student-fp16", "full", QWEN_05B),
        ("student-qat-3bit-full", "full", QWEN_05B),
    ],
)
def test_the_scored_model_is_the_student_only_for_distillation(
    name: str, mode: RunMode, expected: str
) -> None:
    assert scored_model(_resolve(name, mode).variant).model_id == expected


@pytest.mark.parametrize(("mode", "expected"), [("full", QWEN_3B), ("smoke", SMOLLM2)])
def test_the_teacher_is_the_model_id(mode: RunMode, expected: str) -> None:
    assert teacher_model(_resolve("distill-hardlabel", mode).variant).model_id == expected


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("kd_temperature: 1.0", "kd_temperature: 2.0", "kd_temperature 2.0.*frontier recalibrate"),
        ("epochs: 1", "epochs: 2", "epochs 2.*one pass"),
        ("family: distill", "family: baseline", "has a distill block"),
    ],
)
def test_a_block_the_cache_cannot_train_is_refused_at_resolve(
    tmp_path: Path, old: str, new: str, match: str
) -> None:
    text = SOFT.read_text(encoding="utf-8").replace(old, new)
    with pytest.raises(ValueError, match=match):
        resolve_config(_variant_file(tmp_path, text), config_root=CONFIG_ROOT)


def test_a_distill_family_without_a_block_is_refused(tmp_path: Path) -> None:
    text = (CONFIG_ROOT / "variants" / "fp16.yaml").read_text(encoding="utf-8")
    path = _variant_file(tmp_path, text.replace("family: baseline", "family: distill"))
    with pytest.raises(ValueError, match="has no distill block"):
        resolve_config(path, config_root=CONFIG_ROOT)


def test_a_distill_variant_needs_a_student(tmp_path: Path) -> None:
    text = SOFT.read_text(encoding="utf-8").replace(f"student_id: {QWEN_05B}", "student_id: null")
    with pytest.raises(ValueError, match=r"no model\.student_id"):
        resolve_config(_variant_file(tmp_path, text), config_root=CONFIG_ROOT)
    variant = _resolve("distill-softlabel-topk").variant
    studentless = replace(variant, model=replace(variant.model, student_id=None))
    with pytest.raises(ValueError, match=r"no model\.student_id"):
        require_distill_spec(studentless)
    with pytest.raises(ValueError, match="no distill block"):
        require_distill_spec(replace(variant, distill=None))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (f"student_id: {QWEN_05B}", "student_id: Qwen/Qwen2.5-1.5B-Instruct"),
        ("learning_rate: 1.0e-5", "learning_rate: 2.0e-5"),
    ],
)
def test_what_changes_the_student_changes_the_config_hash(
    tmp_path: Path, old: str, new: str
) -> None:
    text = SOFT.read_text(encoding="utf-8")
    assert old in text
    shipped = resolve_config(_variant_file(tmp_path, text), config_root=CONFIG_ROOT).config_hash
    edited = resolve_config(
        _variant_file(tmp_path, text.replace(old, new)), config_root=CONFIG_ROOT
    ).config_hash
    assert edited != shipped
