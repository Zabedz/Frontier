"""Recipe descriptors (pure) and the modifier build (gated on llmcompressor)."""

from __future__ import annotations

import pytest

from frontier.quantize.recipes import RecipeKind, RecipeSpec, recipe_for, to_modifiers
from frontier.schema import QuantSpec

GROUP_SIZE = 128
SMALL_GROUP = 64
PAIR = 2


def _quant(method: str, *, group_size: int = GROUP_SIZE) -> QuantSpec:
    return QuantSpec(method=method, bit_width=4, group_size=group_size)


def test_recipe_for_maps_the_three_methods() -> None:
    assert recipe_for(_quant("llmcompressor-gptq")).kind == "gptq"
    assert recipe_for(_quant("llmcompressor-awq")).kind == "awq"
    assert recipe_for(_quant("llmcompressor-w8a8", group_size=-1)).kind == "w8a8"


def test_recipe_for_pulls_group_size() -> None:
    assert (
        recipe_for(_quant("llmcompressor-gptq", group_size=SMALL_GROUP)).group_size == SMALL_GROUP
    )


def test_recipe_for_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="no llm-compressor recipe"):
        recipe_for(_quant("bnb-nf4"))


def test_recipe_for_rejects_gguf_method() -> None:
    with pytest.raises(ValueError, match="no llm-compressor recipe"):
        recipe_for(_quant("gguf"))


def test_to_modifiers_gptq_is_single_w4a16() -> None:
    pytest.importorskip("llmcompressor")
    modifiers = to_modifiers(RecipeSpec(kind="gptq", group_size=GROUP_SIZE))
    assert len(modifiers) == 1
    assert modifiers[0].scheme == "W4A16"
    assert modifiers[0].actorder == "weight"


@pytest.mark.parametrize("kind", ["gptq", "awq"])
def test_to_modifiers_refuses_a_group_size_the_preset_lacks(kind: RecipeKind) -> None:
    pytest.importorskip("llmcompressor")
    with pytest.raises(ValueError, match="disagrees with the"):
        to_modifiers(RecipeSpec(kind=kind, group_size=SMALL_GROUP))


def test_to_modifiers_awq_pairs_scale_learning_with_asym_quant() -> None:
    pytest.importorskip("llmcompressor")
    modifiers = to_modifiers(RecipeSpec(kind="awq", group_size=GROUP_SIZE))
    assert len(modifiers) == PAIR
    # The 0.12 awq.AWQModifier shim returns a list holding an empty quantizer.
    assert type(modifiers[0]).__module__.startswith("llmcompressor.modifiers.transform")
    assert modifiers[1].scheme == "W4A16_ASYM"


def test_to_modifiers_w8a8_runs_smoothquant_then_gptq() -> None:
    pytest.importorskip("llmcompressor")
    modifiers = to_modifiers(RecipeSpec(kind="w8a8", group_size=-1))
    assert len(modifiers) == PAIR
    assert modifiers[1].scheme == "W8A8"
