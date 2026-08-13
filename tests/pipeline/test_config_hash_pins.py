"""The config hashes of every banked variant, pinned.

Six rows are already in the result store. Their ``config_hash`` keys the resume-skip and
names their predictions sidecar, so a config edit that moves one of these hashes orphans a
banked row: a re-run stops matching ``_done_seeds`` and appends a duplicate instead of
skipping. That is cheap to do by accident, because the hash covers the whole merged config
and a key added to ``configs/base.yaml`` reaches every variant.

These pins fail loudly when that happens. A deliberate change to one of these variants
means updating its pin here in the same commit, which is the point: the change becomes a
decision in the diff.
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
    """bitsandbytes NF4 and the GGUF k-quants derive from the weights alone.

    Giving them a seed would put a value in the hash that changes nothing about the
    checkpoint, and would move the pins above.
    """
    for variant in ("int4-nf4", "gguf-q4_k_m"):
        resolved = resolve_config(
            CONFIG_ROOT / "variants" / f"{variant}.yaml", config_root=CONFIG_ROOT
        )
        assert resolved.variant.quant is not None
        assert resolved.variant.quant.calibration_seed is None


def test_the_shared_defaults_carry_no_quant_block() -> None:
    """``configs/base.yaml`` must stay clear of quant keys.

    Every variant overlays base, so a quant key there would enter every merged config and
    move every hash in ``BANKED_HASHES`` at once.
    """
    resolved = resolve_config(CONFIG_ROOT / "variants" / "fp16.yaml", config_root=CONFIG_ROOT)
    assert resolved.raw.get("quant") is None
