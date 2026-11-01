"""Backend dispatch: provider selection and latency-probe mapping, no GPU touched.

The full-mode providers are lazy, so constructing one asserts dispatch on a laptop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frontier.backends.hf import HFLogitProvider
from frontier.backends.llama_cpp import LlamaCppLogitProvider
from frontier.backends.registry import build_latency_probe, build_provider
from frontier.backends.torchao import ADAPTER_FILE, TorchaoLogitProvider
from frontier.backends.vllm import VllmLogitProvider
from frontier.distill.paths import STUDENT_METADATA_FILE, student_path
from frontier.latency.native import NativeLlamaCppLatency, NativeVllmLatency
from frontier.latency.rig import default_latency
from frontier.pipeline.config import resolve_config
from frontier.quantize.paths import checkpoint_path
from frontier.schema import RunMode, VariantConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
CHECKPOINTS = Path("/workspace/checkpoints")
QWEN = "Qwen/Qwen2.5-3B-Instruct"


def _resolve(name: str) -> tuple[VariantConfig, dict[str, object]]:
    resolved = resolve_config(CONFIG_ROOT / "variants" / f"{name}.yaml", config_root=CONFIG_ROOT)
    return resolved.variant, dict(resolved.backend)


@pytest.mark.parametrize("name", ["fp16", "int4-nf4", "int4-gptq", "gguf-q4_k_m"])
def test_smoke_returns_hf_for_every_gpu_only_backend(name: str) -> None:
    variant, backend = _resolve(name)
    provider = build_provider(
        variant, backend, device="cpu", mode="smoke", checkpoints_root=CHECKPOINTS
    )
    assert type(provider) is HFLogitProvider


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_torchao_ptq_quantises_on_load_in_every_mode(mode: RunMode) -> None:
    variant, backend = _resolve("ptq-3bit-torchao")
    provider = build_provider(
        variant, backend, device="cpu", mode=mode, checkpoints_root=CHECKPOINTS
    )
    assert isinstance(provider, TorchaoLogitProvider)
    assert (provider.bit_width, provider.group_size) == (3, 32)
    assert provider.model_id == QWEN


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


def test_full_vllm_quant_serves_each_seeds_checkpoint() -> None:
    variant, backend = _resolve("int4-gptq")
    provider = build_provider(
        variant, backend, device="cuda", mode="full", checkpoints_root=CHECKPOINTS, seed=2
    )
    assert isinstance(provider, VllmLogitProvider)
    assert provider.model == str(checkpoint_path(variant, backend, root=CHECKPOINTS, seed=2))
    assert provider.model.endswith("-s2")


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


def test_torchao_full_qat_serves_each_seeds_trained_checkpoint() -> None:
    variant, backend = _resolve("student-qat-3bit-full")
    provider = build_provider(
        variant, backend, device="cuda", mode="full", checkpoints_root=CHECKPOINTS, seed=2
    )
    assert isinstance(provider, TorchaoLogitProvider)
    assert provider.model_id == str(checkpoint_path(variant, backend, root=CHECKPOINTS, seed=2))
    assert provider.adapter is None
    assert (provider.bit_width, provider.group_size) == (3, 32)


def test_torchao_lora_qat_serves_the_base_with_each_seeds_adapter() -> None:
    variant, backend = _resolve("qat-3bit-lora")
    provider = build_provider(
        variant, backend, device="cuda", mode="full", checkpoints_root=CHECKPOINTS, seed=2
    )
    assert isinstance(provider, TorchaoLogitProvider)
    assert provider.model_id == variant.model.model_id
    checkpoint = checkpoint_path(variant, backend, root=CHECKPOINTS, seed=2)
    assert provider.adapter == checkpoint / ADAPTER_FILE
    assert (provider.bit_width, provider.group_size) == (3, 32)


def test_latency_probe_maps_backends() -> None:
    _, hf_backend = _resolve("fp16")
    _, vllm_backend = _resolve("int4-gptq")
    _, gguf_backend = _resolve("gguf-q4_k_m")
    assert build_latency_probe(hf_backend, mode="full") is default_latency
    assert isinstance(build_latency_probe(vllm_backend, mode="full"), NativeVllmLatency)
    assert isinstance(build_latency_probe(gguf_backend, mode="full"), NativeLlamaCppLatency)
    assert build_latency_probe(vllm_backend, mode="smoke") is default_latency
    _, torchao_backend = _resolve("ptq-3bit-torchao")
    assert build_latency_probe(torchao_backend, mode="full") is default_latency


def _trained_student(variant: VariantConfig, root: Path, *, seed: int, capped: bool) -> Path:
    path = student_path(variant, root=root, seed=seed)
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    metadata = {"capped": capped, "budget_tokens": 16_384 if capped else 30_000_000}
    (path / STUDENT_METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_a_distill_variant_serves_each_seeds_trained_student(tmp_path: Path) -> None:
    variant, backend = _resolve("distill-softlabel-topk")
    trained = _trained_student(variant, tmp_path, seed=2, capped=False)
    provider = build_provider(
        variant, backend, device="cuda", mode="full", checkpoints_root=tmp_path, seed=2
    )
    assert isinstance(provider, HFLogitProvider)
    assert provider.model_id == str(trained)
    assert provider.weight_dtype == "fp16"
    assert QWEN not in provider.model_id


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_a_distill_variant_without_its_student_raises(tmp_path: Path, mode: RunMode) -> None:
    variant, backend = _resolve("distill-hardlabel")
    with pytest.raises(FileNotFoundError, match="no trained student"):
        build_provider(variant, backend, device="cpu", mode=mode, checkpoints_root=tmp_path)


def test_a_capped_student_is_not_served_in_full_mode(tmp_path: Path) -> None:
    variant, backend = _resolve("distill-corpus")
    _trained_student(variant, tmp_path, seed=0, capped=True)
    with pytest.raises(ValueError, match="capped budget"):
        build_provider(variant, backend, device="cuda", mode="full", checkpoints_root=tmp_path)


def test_a_distill_variant_at_another_dtype_is_refused(tmp_path: Path) -> None:
    variant, backend = _resolve("distill-corpus")
    _trained_student(variant, tmp_path, seed=0, capped=False)
    with pytest.raises(ValueError, match="saved and scored in"):
        build_provider(
            variant,
            {**backend, "weight_dtype": "nf4"},
            device="cuda",
            mode="full",
            checkpoints_root=tmp_path,
        )


def test_a_distill_variant_on_another_backend_is_refused(tmp_path: Path) -> None:
    variant, backend = _resolve("distill-corpus")
    with pytest.raises(ValueError, match="served on hf"):
        build_provider(
            variant,
            {**backend, "inference_backend": "torchao"},
            device="cpu",
            mode="full",
            checkpoints_root=tmp_path,
        )
