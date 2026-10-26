"""The torchao Track-A backend: a Hugging Face model whose linear layers torchao quantises on
load. One config builder serves the PTQ control and QAT, so the two arms differ only in
whether the weights were adapted. ``torch`` and ``torchao`` are imported on first use.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from frontier.backends.hf import DEFAULT_REVISION, HFLogitProvider

# torch exposes sub-byte dtypes torch.int1 .. torch.int7; torchao packs them as intx.
MIN_BITS, MAX_BITS = 1, 7

# Excluded from quantisation, as in the llm-compressor recipes: Qwen ties it to the embedding.
OUTPUT_HEAD = "lm_head"

# QAT+LoRA adapter keys: "<linear>.up" (out x r, scaling folded in), "<linear>.down" (r x in).
ADAPTER_FILE = "adapter.safetensors"
UP, DOWN = "up", "down"


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


def attach_adapter(model: Any, factors: Mapping[str, Any], *, dtype: Any) -> int:
    """Add ``x @ down.T @ up.T`` to each named linear's output; returns how many it adapted.

    This is the function QAT+LoRA trains: the quantised base plus an unquantised adapter.
    Merging the adapter and re-quantising re-rounds the weights and undoes the training.
    ``dtype`` is the compute dtype the factors are stored in. Raises ``ValueError`` for an
    empty adapter, a factor naming no linear, an unpaired or misshapen factor, or a layer
    that already carries an adapter; nothing is attached when any layer fails.
    """
    import torch  # noqa: PLC0415

    if not factors:
        raise ValueError("adapter has no factors")
    modules = dict(model.named_modules())
    checked = []
    for name in sorted({key.rsplit(".", 1)[0] for key in factors}):
        layer = modules.get(name)
        if not isinstance(layer, torch.nn.Linear):
            raise ValueError(f"adapter names {name!r}, which is not a linear layer of the model")
        if hasattr(layer, "adapter_up"):
            raise ValueError(f"{name!r} already carries an adapter")
        up, down = factors.get(f"{name}.{UP}"), factors.get(f"{name}.{DOWN}")
        if up is None or down is None:
            raise ValueError(f"adapter for {name!r} needs both .{UP} and .{DOWN}")
        rank = down.shape[0] if down.ndim > 1 else -1
        if (tuple(up.shape), tuple(down.shape)) != (
            (layer.out_features, rank),
            (rank, layer.in_features),
        ):
            raise ValueError(
                f"adapter for {name!r} has up {tuple(up.shape)} and down {tuple(down.shape)}"
                f" for a {layer.out_features} x {layer.in_features} layer"
            )
        checked.append((layer, up, down))
    for layer, up, down in checked:
        layer.register_buffer("adapter_up", up.to(layer.weight.device, dtype))
        layer.register_buffer("adapter_down", down.to(layer.weight.device, dtype))
        layer.register_forward_hook(_add_low_rank)
    return len(checked)


def _add_low_rank(layer: Any, args: tuple[Any, ...], output: Any) -> Any:
    return output + (args[0] @ layer.adapter_down.T) @ layer.adapter_up.T


class TorchaoLogitProvider(HFLogitProvider):
    """``HFLogitProvider`` with the linear layers quantised to ``int<bit_width>`` after load.

    The forward runs in fp16 on CUDA, the dtype of the HF ``fp16`` reference it is paired
    with, so quantisation is the only difference; fp32 on CPU. torchao dequantises each
    group to that dtype inside the matmul. QAT trained against codes computed from bf16
    (LoRA) or fp32 (full) weights; computing them from fp16 changes about 1% of them.
    ``adapter`` is a QAT+LoRA ``ADAPTER_FILE``, attached unquantised beside the base.
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
        adapter: Path | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id, device=device, weight_dtype=weight_dtype, revision=revision
        )
        self.bit_width = require_bit_width(bit_width)
        self.group_size = group_size
        self.adapter = adapter
        self._torchao_version = "unknown"

    def _compute_dtype_name(self) -> str:
        return "float32" if self.device == "cpu" else "float16"

    def _prepare(self, model: Any) -> Any:
        import torch  # noqa: PLC0415
        import torchao  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415
        from torchao.quantization import quantize_  # noqa: PLC0415

        quantize_(
            model,
            intx_weight_only_config(self.bit_width, self.group_size),
            filter_fn=quantizes,
        )
        if self.adapter is not None:
            dtype = getattr(torch, self._compute_dtype_name())
            attach_adapter(model, load_file(str(self.adapter)), dtype=dtype)
        self._torchao_version = torchao.__version__
        return model

    @property
    def backend_version(self) -> str:
        """The torchao and ``transformers`` versions, or ``"unknown"`` before load."""
        transformers_version = super().backend_version
        if transformers_version == "unknown":
            return transformers_version
        return f"torchao {self._torchao_version}, transformers {transformers_version}"
