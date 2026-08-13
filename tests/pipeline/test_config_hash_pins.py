"""The config hashes of every banked variant, pinned.

The hash keys the resume-skip and the sidecar name, so moving one orphans a banked row and
a re-run appends a duplicate. A deliberate change updates its pin in the same commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frontier.pipeline.config import resolve_config

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"

BANKED_HASHES = {
    "fp16": "db13d226aa43c7f9ff3fab04c3865175e4273008ff066871936fdcd67dd31ed6",
    "int4-nf4": "faececdb52b888e052785cddb1c09c21d0703cbd4209e26efc352859fb0dd5b8",
    "int8-weightonly": "fc362e08f7523c800205fae0affa50fd2002264e4b3891da28cab61868289d12",
    "gguf-q4_k_m": "1ba8ed66657cf44a3b6186b14da8f634f8225576a413f01b114e44cd0616f9d9",
    "gguf-q5_k_m": "28012fe8322f7e30cefef5cbce0056ca07defafde2b941e15286bc129012111d",
    "gguf-q8_0": "7941753da2812f671a3f621bf86b22a56df2a7a846db5c5c4a0e4096eac626b1",
}

CALIBRATING_VARIANTS = (
    "int4-gptq",
    "int4-gptq-ood",
    "int4-awq",
    "int4-awq-ood",
    "int8-w8a8",
)


@pytest.mark.parametrize(("variant", "expected"), sorted(BANKED_HASHES.items()))
def test_banked_variant_config_hash_is_unchanged(variant: str, expected: str) -> None:
    resolved = resolve_config(CONFIG_ROOT / "variants" / f"{variant}.yaml", config_root=CONFIG_ROOT)
    assert resolved.config_hash == expected


@pytest.mark.parametrize("variant", CALIBRATING_VARIANTS)
def test_a_variant_naming_a_calibration_corpus_also_names_its_seed(variant: str) -> None:
    resolved = resolve_config(CONFIG_ROOT / "variants" / f"{variant}.yaml", config_root=CONFIG_ROOT)
    quant = resolved.variant.quant
    assert quant is not None
    assert quant.calibration_corpus != "none"
    assert quant.calibration_seed is not None


def test_a_data_free_variant_carries_no_calibration_seed() -> None:
    """NF4 and the GGUF k-quants derive from the weights alone, so a seed would only move pins."""
    for variant in ("int4-nf4", "gguf-q4_k_m"):
        resolved = resolve_config(
            CONFIG_ROOT / "variants" / f"{variant}.yaml", config_root=CONFIG_ROOT
        )
        assert resolved.variant.quant is not None
        assert resolved.variant.quant.calibration_seed is None


def test_the_shared_defaults_carry_no_quant_block() -> None:
    """A quant key in ``configs/base.yaml`` would enter every merged config and move every pin."""
    resolved = resolve_config(CONFIG_ROOT / "variants" / "fp16.yaml", config_root=CONFIG_ROOT)
    assert resolved.raw.get("quant") is None
