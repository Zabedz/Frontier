"""The torchao Track-A backend: a Hugging Face model whose linear layers torchao quantises on
load. One config builder serves the PTQ control and QAT, so the two arms differ only in
whether the weights were adapted. ``torch`` and ``torchao`` are imported on first use.
"""

from __future__ import annotations

from typing import Any

from frontier.backends.hf import DEFAULT_REVISION, HFLogitProvider

# torch exposes sub-byte dtypes torch.int1 .. torch.int7; torchao packs them as intx.
MIN_BITS, MAX_BITS = 1, 7

# Excluded from quantisation, as in the llm-compressor recipes: Qwen ties it to the embedding.
OUTPUT_HEAD = "lm_head"


def require_bit_width(bit_width: int) -> int:
    """``bit_width`` when torch has an ``int<bit_width>`` dtype, else ``ValueError``."""
    if not MIN_BITS <= bit_width <= MAX_BITS:
        raise ValueError(
            f"torchao intx needs a bit width in {MIN_BITS}..{MAX_BITS}, got {bit_width}"
        )
    return bit_width


def intx_weight_only_config(bit_width: int, group_size: int) -> Any:
    """torchao's group-wise ``int<bit_width>`` weight-only config."""
    import torch  # noqa: PLC0415
    from torchao.quantization import IntxWeightOnlyConfig  # noqa: PLC0415
    from torchao.quantization.granularity import PerGroup  # noqa: PLC0415

    weight_dtype = getattr(torch, f"int{require_bit_width(bit_width)}")
    return IntxWeightOnlyConfig(weight_dtype=weight_dtype, granularity=PerGroup(group_size))


def quantizes(module: Any, fqn: str) -> bool:
    """``quantize_`` filter: every ``nn.Linear`` except the output head."""
    import torch  # noqa: PLC0415

    return isinstance(module, torch.nn.Linear) and fqn.rsplit(".", 1)[-1] != OUTPUT_HEAD


class TorchaoLogitProvider(HFLogitProvider):
    """``HFLogitProvider`` with the linear layers quantised to ``int<bit_width>`` after load.

    The forward runs in bf16 on CUDA (Qwen's native dtype) and fp32 on CPU; torchao
    dequantises each group to that dtype inside the matmul.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        weight_dtype: str,
        bit_width: int,
        group_size: int,
        revision: str = DEFAULT_REVISION,
    ) -> None:
        super().__init__(
            model_id=model_id, device=device, weight_dtype=weight_dtype, revision=revision
        )
        self.bit_width = require_bit_width(bit_width)
        self.group_size = group_size
        self._torchao_version = "unknown"

    def _compute_dtype_name(self) -> str:
        return "float32" if self.device == "cpu" else "bfloat16"

    def _prepare(self, model: Any) -> Any:
        import torchao  # noqa: PLC0415
        from torchao.quantization import quantize_  # noqa: PLC0415

        quantize_(
            model,
            intx_weight_only_config(self.bit_width, self.group_size),
            filter_fn=quantizes,
        )
        self._torchao_version = torchao.__version__
        return model

    @property
    def backend_version(self) -> str:
        """The torchao and ``transformers`` versions, or ``"unknown"`` before load."""
        transformers_version = super().backend_version
        if transformers_version == "unknown":
            return transformers_version
        return f"torchao {self._torchao_version}, transformers {transformers_version}"
