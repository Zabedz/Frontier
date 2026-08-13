"""The ``frontier-quantize`` dispatch: producer selection by backend, model-free."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import typer

from frontier.pipeline.config import resolve_config
from frontier.quantize import cli
from frontier.schema import VariantConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
CALIBRATION_SEED = 620


def test_produce_dispatches_to_compressed_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved = resolve_config(CONFIG_ROOT / "variants" / "int4-gptq.yaml", config_root=CONFIG_ROOT)
    seen: dict[str, Any] = {}

    def fake(variant: VariantConfig, backend: Mapping[str, Any], *, checkpoints_root: Path) -> Path:
        seen["name"] = variant.name
        seen["backend"] = backend["inference_backend"]
        seen["checkpoints_root"] = checkpoints_root
        seen["calibration_seed"] = variant.quant.calibration_seed if variant.quant else None
        return checkpoints_root / "ckpt"

    monkeypatch.setattr(cli, "produce_compressed_tensors", fake)
    out = cli._produce(resolved.variant, resolved.backend, tmp_path)
    assert out == tmp_path / "ckpt"
    assert seen == {
        "name": "int4-gptq",
        "backend": "vllm",
        "checkpoints_root": tmp_path,
        "calibration_seed": CALIBRATION_SEED,
    }


def test_produce_rejects_a_backend_without_a_producer(tmp_path: Path) -> None:
    resolved = resolve_config(CONFIG_ROOT / "variants" / "fp16.yaml", config_root=CONFIG_ROOT)
    with pytest.raises(typer.BadParameter, match="no producer"):
        cli._produce(resolved.variant, resolved.backend, tmp_path)
