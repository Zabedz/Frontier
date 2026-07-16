"""Backend dispatch: provider selection and latency-probe mapping, no GPU touched.

Full mode constructs the vLLM / llama.cpp providers without loading their engines (the
providers are lazy), so the dispatch is asserted by the returned type and its served path
on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frontier.backends.hf import HFLogitProvider
from frontier.backends.llama_cpp import LlamaCppLogitProvider
from frontier.backends.registry import build_latency_probe, build_provider
from frontier.backends.vllm import VllmLogitProvider
from frontier.latency.native import NativeLlamaCppLatency, NativeVllmLatency
from frontier.latency.rig import default_latency
from frontier.pipeline.config import resolve_config
from frontier.quantize.paths import checkpoint_path
from frontier.schema import VariantConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
CHECKPOINTS = Path("/workspace/checkpoints")
QWEN = "Qwen/Qwen2.5-3B-Instruct"


def _resolve(name: str) -> tuple[VariantConfig, dict[str, object]]:
    resolved = resolve_config(CONFIG_ROOT / "variants" / f"{name}.yaml", config_root=CONFIG_ROOT)
    return resolved.variant, dict(resolved.backend)


@pytest.mark.parametrize(
    "name", ["fp16", "int4-nf4", "int4-gptq", "gguf-q4_k_m", "ptq-3bit-torchao"]
)
def test_smoke_returns_hf_for_every_backend(name: str) -> None:
    variant, backend = _resolve(name)
    provider = build_provider(
        variant, backend, device="cpu", mode="smoke", checkpoints_root=CHECKPOINTS
    )
    assert isinstance(provider, HFLogitProvider)


def test_full_hf_backend_is_hf_provider() -> None:
    variant, backend = _resolve("int4-nf4")
    provider = build_provider(
        variant, backend, device="cuda", mode="full", checkpoints_root=CHECKPOINTS
    )
    assert isinstance(provider, HFLogitProvider)
    assert provider.weight_dtype == "nf4"


def test_full_vllm_quant_serves_the_checkpoint() -> None:
    variant, backend = _resolve("int4-gptq")
    provider = build_provider(
        variant, backend, device="cuda", mode="full", checkpoints_root=CHECKPOINTS
    )
    assert isinstance(provider, VllmLogitProvider)
    assert provider.model == str(checkpoint_path(variant, backend, root=CHECKPOINTS))


def test_full_vllm_fp16_gate_serves_the_base_model() -> None:
    variant, backend = _resolve("fp16-vllm")
    provider = build_provider(
        variant, backend, device="cuda", mode="full", checkpoints_root=CHECKPOINTS
    )
    assert isinstance(provider, VllmLogitProvider)
    assert provider.model == QWEN


def test_full_llama_cpp_serves_the_gguf() -> None:
    variant, backend = _resolve("gguf-q5_k_m")
    provider = build_provider(
        variant, backend, device="cuda", mode="full", checkpoints_root=CHECKPOINTS
    )
    assert isinstance(provider, LlamaCppLogitProvider)
    assert provider.gguf_path == checkpoint_path(variant, backend, root=CHECKPOINTS)


def test_full_torchao_raises_not_implemented() -> None:
    variant, backend = _resolve("ptq-3bit-torchao")
    with pytest.raises(NotImplementedError, match="torchao"):
        build_provider(variant, backend, device="cuda", mode="full", checkpoints_root=CHECKPOINTS)


def test_latency_probe_maps_backends() -> None:
    _, hf_backend = _resolve("fp16")
    _, vllm_backend = _resolve("int4-gptq")
    _, gguf_backend = _resolve("gguf-q4_k_m")
    assert build_latency_probe(hf_backend, mode="full") is default_latency
    assert isinstance(build_latency_probe(vllm_backend, mode="full"), NativeVllmLatency)
    assert isinstance(build_latency_probe(gguf_backend, mode="full"), NativeLlamaCppLatency)
    assert build_latency_probe(vllm_backend, mode="smoke") is default_latency
