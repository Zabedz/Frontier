"""Deterministic checkpoint-path derivation for the produced Track-B variants.

The producers write to exactly the path :func:`checkpoint_path` returns and
``build_provider`` reads from it, so producing and serving agree. The path encodes the
calibration corpus and sample count, so an out-of-domain checkpoint lands beside the
in-domain one without a collision.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from frontier.quantize.recipes import recipe_for
from frontier.schema import VariantConfig

GGUF_QUANT_TYPES: dict[str, str] = {"q4_k_m": "Q4_K_M", "q5_k_m": "Q5_K_M", "q8_0": "Q8_0"}


def model_slug(model_id: str) -> str:
    """The last path segment of a HF id: ``Qwen/Qwen2.5-3B-Instruct`` -> ``Qwen2.5-3B-Instruct``."""
    return model_id.rsplit("/", maxsplit=1)[-1]


def gguf_quant_type(weight_dtype: str) -> str:
    """Map a ``weight_dtype`` to a ``llama-quantize`` type name (``q4_k_m`` -> ``Q4_K_M``)."""
    try:
        return GGUF_QUANT_TYPES[weight_dtype]
    except KeyError:
        raise ValueError(
            f"weight_dtype {weight_dtype!r} is not a GGUF quant type; "
            f"expected one of {sorted(GGUF_QUANT_TYPES)}"
        ) from None


def checkpoint_path(variant: VariantConfig, backend: Mapping[str, Any], *, root: Path) -> Path:
    """Where a variant's produced checkpoint lives, deterministically.

    vLLM lands at ``root/compressed_tensors/<slug>/<kind>-<corpus>-<samples>s-g<group>``
    and llama.cpp at ``root/gguf/<slug>.<weight_dtype>.gguf``. Raises ``ValueError`` for
    ``hf`` and ``torchao``, which quantise the base model in-process and produce no
    checkpoint, and for a vLLM variant with no ``quant`` block, which is the FP16 gate
    serving the base model directly.
    """
    inference_backend = backend["inference_backend"]
    slug = model_slug(variant.model.model_id)
    if inference_backend == "vllm":
        if variant.quant is None:
            raise ValueError(
                f"vllm variant {variant.name!r} has no quant block; the FP16 gate serves the "
                f"base model directly and has no checkpoint"
            )
        recipe = recipe_for(variant.quant)
        name = (
            f"{recipe.kind}-{variant.quant.calibration_corpus}-"
            f"{variant.quant.calibration_samples}s-g{variant.quant.group_size}"
        )
        return root / "compressed_tensors" / slug / name
    if inference_backend == "llama_cpp":
        return root / "gguf" / f"{slug}.{backend['weight_dtype']}.gguf"
    raise ValueError(
        f"backend {inference_backend!r} has no produced checkpoint "
        f"(hf/torchao load or quantise the base model in-process)"
    )
