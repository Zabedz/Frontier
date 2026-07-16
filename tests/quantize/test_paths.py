"""Checkpoint-path derivation and the GGUF quant-type map, from the real configs."""

from __future__ import annotations

from pathlib import Path

import pytest

from frontier.pipeline.config import resolve_config
from frontier.quantize.paths import checkpoint_path, gguf_quant_type, model_slug
from frontier.schema import VariantConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
ROOT = Path("checkpoints")


def _resolve(name: str) -> tuple[VariantConfig, dict[str, object]]:
    resolved = resolve_config(CONFIG_ROOT / "variants" / f"{name}.yaml", config_root=CONFIG_ROOT)
    return resolved.variant, dict(resolved.backend)


def test_model_slug_takes_last_segment() -> None:
    assert model_slug("Qwen/Qwen2.5-3B-Instruct") == "Qwen2.5-3B-Instruct"


def test_compressed_tensors_path_encodes_recipe_and_calibration() -> None:
    variant, backend = _resolve("int4-gptq")
    assert checkpoint_path(variant, backend, root=ROOT) == (
        ROOT / "compressed_tensors" / "Qwen2.5-3B-Instruct" / "gptq-in_domain-512s-g128"
    )


def test_compressed_tensors_path_awq_and_w8a8() -> None:
    awq_variant, awq_backend = _resolve("int4-awq")
    assert checkpoint_path(awq_variant, awq_backend, root=ROOT).name == "awq-in_domain-512s-g128"
    w8a8_variant, w8a8_backend = _resolve("int8-w8a8")
    assert checkpoint_path(w8a8_variant, w8a8_backend, root=ROOT).name == "w8a8-in_domain-512s-g-1"


def test_ood_variant_lands_beside_in_domain_without_collision() -> None:
    in_domain, in_backend = _resolve("int4-gptq")
    ood, ood_backend = _resolve("int4-gptq-ood")
    in_path = checkpoint_path(in_domain, in_backend, root=ROOT)
    ood_path = checkpoint_path(ood, ood_backend, root=ROOT)
    assert in_path != ood_path
    assert in_path.parent == ood_path.parent
    assert ood_path.name == "gptq-ood-512s-g128"


def test_gguf_path_is_single_file_named_by_dtype() -> None:
    variant, backend = _resolve("gguf-q4_k_m")
    assert checkpoint_path(variant, backend, root=ROOT) == (
        ROOT / "gguf" / "Qwen2.5-3B-Instruct.q4_k_m.gguf"
    )


def test_checkpoint_path_raises_for_hf_backend() -> None:
    variant, backend = _resolve("fp16")
    with pytest.raises(ValueError, match="no produced checkpoint"):
        checkpoint_path(variant, backend, root=ROOT)


def test_checkpoint_path_raises_for_torchao_backend() -> None:
    variant, backend = _resolve("ptq-3bit-torchao")
    with pytest.raises(ValueError, match="no produced checkpoint"):
        checkpoint_path(variant, backend, root=ROOT)


def test_checkpoint_path_raises_for_vllm_without_quant() -> None:
    variant, backend = _resolve("fp16-vllm")
    with pytest.raises(ValueError, match="no quant block"):
        checkpoint_path(variant, backend, root=ROOT)


def test_gguf_quant_type_map_and_unknown() -> None:
    assert gguf_quant_type("q4_k_m") == "Q4_K_M"
    assert gguf_quant_type("q5_k_m") == "Q5_K_M"
    assert gguf_quant_type("q8_0") == "Q8_0"
    with pytest.raises(ValueError, match="not a GGUF quant type"):
        gguf_quant_type("nf4")
