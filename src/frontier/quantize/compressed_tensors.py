"""The llm-compressor oneshot producer: base model -> compressed-tensors checkpoint.

GPU-only. One ``oneshot`` call runs the variant's recipe over the calibration set and
saves the compressed safetensors, tokenizer, recipe, and config under ``checkpoint_path``,
which is where ``build_provider`` later points vLLM.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from frontier.quantize.calibration import build_calibration_dataset
from frontier.quantize.paths import checkpoint_path
from frontier.quantize.recipes import recipe_for, to_modifiers
from frontier.schema import VariantConfig

# Fixed across variants, leaving the corpus and the draw as the only calibration
# difference between two of them, and both of those land in the config hash.
CALIB_SEQ_LEN = 2048

_COMPLETION_MARKERS = ("config.json", "recipe.yaml")


def produce_compressed_tensors(
    variant: VariantConfig,
    backend: Mapping[str, Any],
    *,
    checkpoints_root: Path,
) -> Path:
    """Quantise ``variant`` with llm-compressor and write a compressed-tensors checkpoint.

    Idempotent: returns early when the checkpoint already carries a ``config.json`` and a
    ``recipe.yaml``. The calibration seed comes from the config, so the draw is covered by
    the config hash and two checkpoints built from different draws stay distinguishable in
    the store. Raises ``ValueError`` for a variant with no ``quant`` block, or one that
    calibrates with no ``calibration_seed``.
    """
    if variant.quant is None:
        raise ValueError(f"variant {variant.name!r} has no quant block to compress")
    if variant.quant.calibration_corpus != "none" and variant.quant.calibration_seed is None:
        raise ValueError(
            f"variant {variant.name!r} calibrates on the {variant.quant.calibration_corpus!r} "
            f"corpus but sets no calibration_seed; the draw would go unrecorded"
        )
    out = checkpoint_path(variant, backend, root=checkpoints_root)
    if _is_complete(out):
        return out
    return _run_oneshot(variant, out)  # pragma: no cover


def _is_complete(out: Path) -> bool:
    return out.is_dir() and all((out / marker).exists() for marker in _COMPLETION_MARKERS)


def _run_oneshot(variant: VariantConfig, out: Path) -> Path:  # pragma: no cover
    from llmcompressor import oneshot  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    # Both are narrowed by the caller's guards; the asserts are for the type checker.
    assert variant.quant is not None
    assert variant.quant.calibration_seed is not None
    model = AutoModelForCausalLM.from_pretrained(
        variant.model.model_id, dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(variant.model.model_id)
    dataset = build_calibration_dataset(
        variant.quant.calibration_corpus,
        tokenizer,
        num_samples=variant.quant.calibration_samples,
        max_seq_length=CALIB_SEQ_LEN,
        seed=variant.quant.calibration_seed,
    )
    oneshot(
        model=model,
        dataset=dataset,
        recipe=to_modifiers(recipe_for(variant.quant)),
        num_calibration_samples=variant.quant.calibration_samples,
        max_seq_length=CALIB_SEQ_LEN,
        output_dir=str(out),
    )
    return out
