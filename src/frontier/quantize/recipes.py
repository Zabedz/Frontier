"""llm-compressor recipe descriptors and the modifier build.

``RecipeSpec`` maps a config ``quant.method`` to a recipe kind and group size with no
llm-compressor import, so path derivation and the tests read it on any machine.
``to_modifiers`` imports ``llmcompressor`` lazily; only the GPU producer reaches it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from frontier.schema import QuantSpec

RecipeKind = Literal["gptq", "awq", "w8a8"]

_METHOD_KIND: dict[str, RecipeKind] = {
    "llmcompressor-gptq": "gptq",
    "llmcompressor-awq": "awq",
    "llmcompressor-w8a8": "w8a8",
}

# llm-compressor's default migration strength for decoder-only models: it shifts most of
# the activation outlier scale onto the weights before per-channel int8.
SMOOTHQUANT_STRENGTH = 0.8


@dataclass(frozen=True, slots=True)
class RecipeSpec:
    """A compressed-tensors recipe reduced to what the path and the modifiers need."""

    kind: RecipeKind
    group_size: int
    ignore: tuple[str, ...] = ("lm_head",)


def recipe_for(quant: QuantSpec) -> RecipeSpec:
    """Map ``quant.method`` to a recipe descriptor; a bnb or gguf method has none."""
    try:
        kind = _METHOD_KIND[quant.method]
    except KeyError:
        raise ValueError(
            f"quant method {quant.method!r} has no llm-compressor recipe; "
            f"expected one of {sorted(_METHOD_KIND)}"
        ) from None
    return RecipeSpec(kind=kind, group_size=quant.group_size)


def to_modifiers(spec: RecipeSpec) -> list[Any]:
    """Build the llm-compressor modifier list for a descriptor (imports llmcompressor).

    The 4-bit group size comes from the preset scheme, so a config that disagrees is
    refused. GPTQ's ``actorder`` is set to ``"weight"``, the accuracy-recovery setting;
    llm-compressor 0.12 defaults to ``"static"``.
    """
    from llmcompressor.modifiers.quantization import (  # noqa: PLC0415
        GPTQModifier,
        QuantizationModifier,
    )
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier  # noqa: PLC0415
    from llmcompressor.modifiers.transform import AWQModifier  # noqa: PLC0415

    ignore = list(spec.ignore)
    if spec.kind == "gptq":
        _require_preset_group("W4A16", spec.group_size)
        return [GPTQModifier(targets="Linear", scheme="W4A16", actorder="weight", ignore=ignore)]
    if spec.kind == "awq":
        _require_preset_group("W4A16_ASYM", spec.group_size)
        return [
            AWQModifier(),
            QuantizationModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=ignore),
        ]
    return [
        SmoothQuantModifier(smoothing_strength=SMOOTHQUANT_STRENGTH),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=ignore),
    ]


def _require_preset_group(scheme: str, group_size: int) -> None:
    from compressed_tensors.quantization import preset_name_to_scheme  # noqa: PLC0415

    preset = preset_name_to_scheme(scheme, ["Linear"]).weights.group_size
    if preset != group_size:
        raise ValueError(
            f"quant.group_size {group_size} disagrees with the {scheme} preset's {preset}; "
            "the modifier takes its group size from the preset"
        )
