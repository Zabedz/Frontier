"""Where the cache and the students live, and the serving-side completeness checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frontier.distill.paths import (
    STUDENT_METADATA_FILE,
    cache_dir,
    require_student,
    student_markers,
    student_path,
)
from frontier.pipeline.config import resolve_config
from frontier.schema import RunMode, VariantConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
ROOT = Path("/workspace/checkpoints")
ARMS = ("distill-softlabel-topk", "distill-corpus", "distill-hardlabel")
SHIPPED_CACHE = "distill_cache/Qwen2.5-3B-Instruct/fineweb_edu-30000000t-L1024-k64"


def _variant(name: str, *, mode: RunMode = "full", profile: str | None = None) -> VariantConfig:
    return resolve_config(
        CONFIG_ROOT / "variants" / f"{name}.yaml",
        eval_profile=profile,
        mode=mode,
        config_root=CONFIG_ROOT,
    ).variant


@pytest.mark.parametrize("mode", ["full", "smoke"])
def test_the_three_arms_share_one_cache(mode: RunMode) -> None:
    paths = {cache_dir(_variant(name, mode=mode), root=ROOT) for name in ARMS}
    assert len(paths) == 1


def test_the_shipped_cache_path_names_teacher_corpus_budget_length_and_k() -> None:
    assert cache_dir(_variant("distill-corpus"), root=ROOT) == ROOT / SHIPPED_CACHE


def test_a_student_path_is_per_variant_and_seed_under_the_student_slug() -> None:
    variant = _variant("distill-softlabel-topk")
    assert student_path(variant, root=ROOT) == (
        ROOT / "distill" / "Qwen2.5-0.5B-Instruct" / "distill-softlabel-topk"
    )
    assert student_path(variant, root=ROOT, seed=2).name == "distill-softlabel-topk-s2"


def test_the_student_path_ignores_the_eval_profile() -> None:
    """The pod job trains seed 0 under one profile and seeds 1-2 under another."""
    one = _variant("distill-softlabel-topk", profile="full-mmlu")
    three = _variant("distill-softlabel-topk", profile="full-mmlu-seeds")
    assert student_path(one, root=ROOT) == student_path(three, root=ROOT)


def _finished(path: Path, *, capped: bool) -> None:
    path.mkdir(parents=True)
    for marker in student_markers():
        (path / marker).write_text("{}", encoding="utf-8")
    metadata = {"capped": capped, "budget_tokens": 16_384 if capped else 30_000_000}
    (path / STUDENT_METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")


@pytest.mark.parametrize("mode", ["full", "smoke"])
def test_a_missing_student_names_its_path_and_producer(tmp_path: Path, mode: RunMode) -> None:
    variant = _variant("distill-hardlabel", mode=mode)
    with pytest.raises(FileNotFoundError, match="frontier-quantize run") as caught:
        require_student(variant, root=tmp_path, seed=1, mode=mode)
    assert str(student_path(variant, root=tmp_path, seed=1)) in str(caught.value)


def test_a_finished_student_is_served(tmp_path: Path) -> None:
    variant = _variant("distill-hardlabel")
    _finished(student_path(variant, root=tmp_path, seed=1), capped=False)
    assert require_student(variant, root=tmp_path, seed=1, mode="full") == student_path(
        variant, root=tmp_path, seed=1
    )


def test_a_capped_student_is_refused_in_full_mode_only(tmp_path: Path) -> None:
    variant = _variant("distill-hardlabel")
    _finished(student_path(variant, root=tmp_path), capped=True)
    with pytest.raises(ValueError, match="capped budget"):
        require_student(variant, root=tmp_path, seed=0, mode="full")
    assert require_student(variant, root=tmp_path, seed=0, mode="smoke").is_dir()
