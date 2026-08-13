"""The compressed-tensors producer's CPU-checkable parts, with no GPU ``oneshot`` run.

The completion guard keeps a re-run of the batch driver from repeating a calibration pass.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from frontier.pipeline.config import resolve_config
from frontier.quantize.compressed_tensors import _is_complete, produce_compressed_tensors
from frontier.quantize.paths import checkpoint_path

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def _complete_checkpoint(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "recipe.yaml").write_text("recipe: {}", encoding="utf-8")
    return root


def test_is_complete_requires_config_and_recipe(tmp_path: Path) -> None:
    assert _is_complete(_complete_checkpoint(tmp_path / "done"))
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "config.json").write_text("{}", encoding="utf-8")
    assert not _is_complete(partial)
    assert not _is_complete(tmp_path / "missing")


def test_producer_returns_early_for_a_complete_checkpoint(tmp_path: Path) -> None:
    resolved = resolve_config(CONFIG_ROOT / "variants" / "int4-gptq.yaml", config_root=CONFIG_ROOT)
    out = checkpoint_path(resolved.variant, resolved.backend, root=tmp_path)
    _complete_checkpoint(out)
    # A re-run must not reach the GPU ``oneshot``, which would import llmcompressor.
    result = produce_compressed_tensors(
        resolved.variant, resolved.backend, checkpoints_root=tmp_path
    )
    assert result == out


def test_producer_rejects_a_variant_without_quant(tmp_path: Path) -> None:
    resolved = resolve_config(CONFIG_ROOT / "variants" / "fp16-vllm.yaml", config_root=CONFIG_ROOT)
    with pytest.raises(ValueError, match="no quant block"):
        produce_compressed_tensors(resolved.variant, resolved.backend, checkpoints_root=tmp_path)


def test_produce_rejects_a_calibrating_variant_with_no_seed(tmp_path: Path) -> None:
    """The draw has to be recorded, or two different checkpoints hash identically."""
    resolved = resolve_config(CONFIG_ROOT / "variants" / "int4-gptq.yaml", config_root=CONFIG_ROOT)
    assert resolved.variant.quant is not None
    seedless = replace(
        resolved.variant, quant=replace(resolved.variant.quant, calibration_seed=None)
    )
    with pytest.raises(ValueError, match="no calibration_seed"):
        produce_compressed_tensors(seedless, resolved.backend, checkpoints_root=tmp_path)
