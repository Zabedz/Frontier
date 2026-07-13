"""Config merge precedence, hash stability, schema validation, and typing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from frontier.pipeline.config import config_hash, deep_merge, resolve_config

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
FP16 = CONFIG_ROOT / "variants" / "fp16.yaml"
SMOKE_SUBSET = 50
FULL_SUBSET = 2000


def test_smoke_precedence_overlays_model_and_eval() -> None:
    resolved = resolve_config(FP16, mode="smoke", config_root=CONFIG_ROOT)
    assert resolved.variant.model.model_id == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert resolved.eval_spec.subset_size == SMOKE_SUBSET
    assert resolved.eval_spec.seeds == (0,)
    assert resolved.variant.latency.batch_sizes == (1,)
    assert resolved.variant.name == "fp16"
    assert resolved.variant.family == "baseline"
    assert resolved.variant.track == "A"
    assert resolved.backend["inference_backend"] == "hf"


def test_full_mode_uses_base_defaults() -> None:
    resolved = resolve_config(FP16, mode="full", config_root=CONFIG_ROOT)
    assert resolved.eval_spec.subset_size == FULL_SUBSET
    assert resolved.variant.model.model_id == "Qwen/Qwen2.5-3B-Instruct"


def test_eval_profile_overlays_task() -> None:
    resolved = resolve_config(
        FP16, eval_profile="secondary-arc", mode="full", config_root=CONFIG_ROOT
    )
    assert resolved.eval_spec.task_name == "arc_challenge"
    assert resolved.eval_spec.labels == "raw"


def test_deep_merge_replaces_lists_and_merges_dicts() -> None:
    base: dict[str, Any] = {"eval": {"seeds": [0, 1, 2], "subset_size": FULL_SUBSET}, "top": 1}
    override: dict[str, Any] = {"eval": {"seeds": [0]}, "extra": 2}
    original_base = {"eval": {"seeds": [0, 1, 2], "subset_size": FULL_SUBSET}, "top": 1}
    original_override = {"eval": {"seeds": [0]}, "extra": 2}

    merged = deep_merge(base, override)
    assert merged["eval"]["seeds"] == [0]
    assert merged["eval"]["subset_size"] == base["eval"]["subset_size"]
    assert merged["top"] == base["top"]
    assert merged["extra"] == override["extra"]
    assert base == original_base
    assert override == original_override


def test_hash_is_stable_and_mode_sensitive() -> None:
    a = resolve_config(FP16, mode="smoke", config_root=CONFIG_ROOT)
    b = resolve_config(FP16, mode="smoke", config_root=CONFIG_ROOT)
    full = resolve_config(FP16, mode="full", config_root=CONFIG_ROOT)
    assert a.config_hash == b.config_hash
    assert a.config_hash != full.config_hash


def test_hash_is_key_order_independent() -> None:
    forward = {"a": 1, "b": {"x": 1, "y": 2}}
    reversed_order = {"b": {"y": 2, "x": 1}, "a": 1}
    assert config_hash(forward) == config_hash(reversed_order)


def test_resolved_smoke_config_validates() -> None:
    resolved = resolve_config(FP16, mode="smoke", config_root=CONFIG_ROOT)
    jsonschema.validate(resolved.raw, _schema())


def test_unknown_key_fails_validation(tmp_path: Path) -> None:
    _reject(tmp_path, "name: x\nfamily: baseline\ntrack: A\nbogus: 1\n")


def test_bad_family_enum_fails_validation(tmp_path: Path) -> None:
    _reject(tmp_path, "name: x\nfamily: not_a_family\ntrack: A\n")


def test_missing_track_fails_validation(tmp_path: Path) -> None:
    _reject(tmp_path, "name: x\nfamily: baseline\n")


def test_typed_views() -> None:
    resolved = resolve_config(FP16, mode="smoke", config_root=CONFIG_ROOT)
    assert isinstance(resolved.eval_spec.seeds, tuple)
    assert resolved.variant.quant is None
    assert resolved.eval_spec is resolved.variant.eval


def _schema() -> dict[str, object]:
    with (CONFIG_ROOT / "schema" / "variant.schema.json").open() as handle:
        loaded: dict[str, object] = json.load(handle)
    return loaded


def _reject(tmp_path: Path, variant_yaml: str) -> None:
    variant = tmp_path / "variant.yaml"
    variant.write_text(variant_yaml)
    with pytest.raises(jsonschema.ValidationError):
        resolve_config(variant, mode="full", config_root=CONFIG_ROOT)
