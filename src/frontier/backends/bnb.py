"""The bitsandbytes weight-dtype mapping, a pure function of the config.

``weight_dtype`` in a Track-A config is either a torch dtype name (``fp16``, ``bf16``)
or a bitsandbytes quantisation scheme (``nf4``, ``int8``). This module is the single
place that knows which strings mean bitsandbytes and what ``BitsAndBytesConfig`` kwargs
each one wants. It builds a plain dict, so the mapping is unit-tested on CPU without
importing transformers or bitsandbytes; ``backends.hf`` turns the dict into a real
``BitsAndBytesConfig`` on the CUDA load path.
"""

from __future__ import annotations

from typing import Any

BNB_WEIGHT_DTYPES = frozenset({"nf4", "int8"})

# bf16 is the Qwen family's native compute dtype, so the NF4 dequant-then-matmul path
# and the LLM.int8 fp path both run in bf16. Named once here so the choice is not a
# magic string spread across the load code.
BNB_COMPUTE_DTYPE = "bfloat16"


def is_bnb_dtype(weight_dtype: str) -> bool:
    """True when ``weight_dtype`` names a bitsandbytes quantisation, not a torch dtype."""
    return weight_dtype in BNB_WEIGHT_DTYPES


def resolve_bnb_compute_dtype(weight_dtype: str) -> str:
    """The torch dtype name bitsandbytes runs the activation path in for this scheme.

    Both bnb schemes keep an fp/bf16 activation path (NF4 dequantises to it, LLM.int8
    mixes an fp16 outlier path), so the compute dtype is ``bfloat16`` for the Qwen
    family regardless of the scheme. Raises ``ValueError`` for a non-bnb dtype, so the
    bnb load branch cannot be reached with an ordinary torch dtype by mistake.
    """
    if not is_bnb_dtype(weight_dtype):
        raise ValueError(f"weight_dtype {weight_dtype!r} is not a bitsandbytes dtype")
    return BNB_COMPUTE_DTYPE


def bnb_config_kwargs(weight_dtype: str, *, compute_dtype: str) -> dict[str, Any]:
    """The ``BitsAndBytesConfig`` kwargs for a bnb ``weight_dtype``.

    ``nf4`` maps to 4-bit NF4 with double quantisation and the given fp/bf16 compute
    dtype; ``int8`` maps to LLM.int8 weight-only (the compute dtype does not apply). The
    ``compute_dtype`` here is a torch dtype *name*; the CUDA caller resolves it to a real
    ``torch`` dtype when constructing ``BitsAndBytesConfig``. Raises ``ValueError`` for a
    non-bnb dtype so a wiring mistake fails loudly rather than loading unquantised.
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
