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

    GPTQ's ``actorder`` is left at the current release's ``"weight"`` default, which is
    the accuracy-recovery setting.
    """
    from llmcompressor.modifiers.awq import AWQModifier  # noqa: PLC0415
    from llmcompressor.modifiers.quantization import (  # noqa: PLC0415
        GPTQModifier,
        QuantizationModifier,
    )
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier  # noqa: PLC0415

    ignore = list(spec.ignore)
    if spec.kind == "gptq":
        return [
            GPTQModifier(
                targets="Linear", scheme="W4A16", group_size=spec.group_size, ignore=ignore
            )
        ]
    if spec.kind == "awq":
        return [
            AWQModifier(),
            QuantizationModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=ignore),
        ]
    return [
        SmoothQuantModifier(smoothing_strength=SMOOTHQUANT_STRENGTH),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=ignore),
    ]
