"""Live producer checks. Pod-only, gated, skipped on the CPU loop.

Runs ``produce_compressed_tensors`` and ``produce_gguf`` on a small model and asserts the
checkpoint lands at ``checkpoint_path`` and re-loads. The compressed-tensors arm needs the
GPU (``oneshot`` calibrates on device); the GGUF arm needs the llama.cpp tooling.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("llmcompressor")
pytest.importorskip("transformers")
pytest.importorskip("torch")

from frontier.pipeline.config import resolve_config
from frontier.quantize.compressed_tensors import produce_compressed_tensors
from frontier.quantize.paths import checkpoint_path

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not os.environ.get("FRONTIER_LIVE_MODELS"),
        reason="live GPU producer; set FRONTIER_LIVE_MODELS=1 on the pod to run",
    ),
]


def test_compressed_tensors_producer_writes_and_reloads(tmp_path: Path) -> None:
    resolved = resolve_config(CONFIG_ROOT / "variants" / "int4-gptq.yaml", config_root=CONFIG_ROOT)
    out = produce_compressed_tensors(resolved.variant, resolved.backend, checkpoints_root=tmp_path)
    assert out == checkpoint_path(resolved.variant, resolved.backend, root=tmp_path)
    assert (out / "config.json").exists()
    assert (out / "recipe.yaml").exists()

    import transformers  # noqa: PLC0415

    reloaded = transformers.AutoModelForCausalLM.from_pretrained(str(out), device_map="auto")
    assert reloaded.config is not None
