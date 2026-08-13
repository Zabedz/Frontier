"""The bitsandbytes weight-dtype mapping. It builds plain ``BitsAndBytesConfig`` kwargs, so
the mapping is unit-tested on CPU without importing transformers or bitsandbytes.
"""

from __future__ import annotations

from typing import Any

BNB_WEIGHT_DTYPES = frozenset({"nf4", "int8"})

# bf16 is the Qwen family's native compute dtype, and both bnb schemes keep an fp/bf16
# activation path (NF4 dequantises to it, LLM.int8 mixes an fp16 outlier path).
BNB_COMPUTE_DTYPE = "bfloat16"


def is_bnb_dtype(weight_dtype: str) -> bool:
    """True when ``weight_dtype`` names a bitsandbytes quantisation."""
    return weight_dtype in BNB_WEIGHT_DTYPES


def resolve_bnb_compute_dtype(weight_dtype: str) -> str:
    """The torch dtype name bitsandbytes runs this scheme's activation path in."""
    if not is_bnb_dtype(weight_dtype):
        raise ValueError(f"weight_dtype {weight_dtype!r} is not a bitsandbytes dtype")
    return BNB_COMPUTE_DTYPE


def bnb_config_kwargs(weight_dtype: str, *, compute_dtype: str) -> dict[str, Any]:
    """The ``BitsAndBytesConfig`` kwargs for a bnb ``weight_dtype``.

    ``nf4`` is 4-bit NF4 with double quantisation; ``int8`` is LLM.int8 weight-only, which
    takes no compute dtype. ``compute_dtype`` is a torch dtype name the CUDA caller resolves.
    """
    if weight_dtype == "nf4":
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": compute_dtype,
        }
    if weight_dtype == "int8":
        return {"load_in_8bit": True}
    raise ValueError(f"weight_dtype {weight_dtype!r} is not a bitsandbytes dtype")
