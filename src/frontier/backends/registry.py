"""Backend dispatch: the logit provider and the latency probe for a variant. Smoke mode
collapses every backend onto the HF provider, since the other stacks are GPU-only.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from frontier.backends.hf import HFLogitProvider
from frontier.backends.llama_cpp import LlamaCppLogitProvider
from frontier.backends.vllm import VllmLogitProvider
from frontier.eval.provider import LogitProvider
from frontier.latency.native import NativeLlamaCppLatency, NativeVllmLatency
from frontier.latency.rig import default_latency
from frontier.quantize.paths import checkpoint_path
from frontier.schema import RunMode, VariantConfig

if TYPE_CHECKING:
    from frontier.pipeline.runner import LatencyProbe


def build_provider(
    variant: VariantConfig,
    backend: Mapping[str, Any],
    *,
    device: str,
    mode: RunMode,
    checkpoints_root: Path,
    seed: int = 0,
) -> LogitProvider:
    """Pick the logit provider for a variant's ``backend.inference_backend``.

    ``hf`` covers the bnb variants too, which the provider branches on off ``weight_dtype``.
    ``vllm`` serves the base model directly for the FP16 gate, which has no ``quant``.
    """
    inference_backend = backend["inference_backend"]
    if mode == "smoke" or inference_backend == "hf":
        return HFLogitProvider(
            model_id=variant.model.model_id,
            device=device,
            weight_dtype=str(backend["weight_dtype"]),
            revision=variant.model.model_revision,
        )
    if inference_backend == "vllm":
        model = (
            str(checkpoint_path(variant, backend, root=checkpoints_root))
            if variant.quant is not None
            else variant.model.model_id
        )
        return VllmLogitProvider(
            model=model,
            tokenizer_id=variant.model.model_id,
            device=device,
            weight_dtype=str(backend["weight_dtype"]),
            seed=seed,
            revision=variant.model.model_revision,
        )
    if inference_backend == "llama_cpp":
        return LlamaCppLogitProvider(
            gguf_path=checkpoint_path(variant, backend, root=checkpoints_root),
            tokenizer_id=variant.model.model_id,
            device=device,
            n_gpu_layers=int(backend["gpu_offload_layers"]),
            weight_dtype=str(backend["weight_dtype"]),
            seed=seed,
            revision=variant.model.model_revision,
        )
    if inference_backend == "torchao":
        raise NotImplementedError("the torchao backend arrives with the QAT work package")
    raise ValueError(f"unknown inference_backend {inference_backend!r}")


def build_latency_probe(backend: Mapping[str, Any], *, mode: RunMode) -> LatencyProbe:
    """The latency probe for a backend.

    Smoke and ``hf`` use the CUDA-event rig. vLLM and llama.cpp models are not
    ``nn.Module``s, so Python-side per-token marking is meaningless and each times itself.
    """
    inference_backend = backend["inference_backend"]
    if mode == "smoke" or inference_backend == "hf":
        return default_latency
    if inference_backend == "vllm":
        return NativeVllmLatency()
    if inference_backend == "llama_cpp":
        return NativeLlamaCppLatency()
    raise ValueError(f"no latency probe for inference_backend {inference_backend!r}")
