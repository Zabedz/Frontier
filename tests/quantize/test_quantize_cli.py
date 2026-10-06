"""The ``frontier-quantize`` dispatch: producer selection by backend, model-free."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from frontier.pipeline.config import resolve_config
from frontier.quantize import cli
from frontier.schema import RunMode, VariantConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
CALIBRATION_SEED = 620


def test_produce_dispatches_to_compressed_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved = resolve_config(CONFIG_ROOT / "variants" / "int4-gptq.yaml", config_root=CONFIG_ROOT)
    seen: dict[str, Any] = {}

    def fake(
        variant: VariantConfig, backend: Mapping[str, Any], *, checkpoints_root: Path, seed: int
    ) -> Path:
        seen["name"] = variant.name
        seen["seed"] = seed
        seen["backend"] = backend["inference_backend"]
        seen["checkpoints_root"] = checkpoints_root
        seen["calibration_seed"] = variant.quant.calibration_seed if variant.quant else None
        return checkpoints_root / "ckpt"

    monkeypatch.setattr(cli, "produce_compressed_tensors", fake)
    out = cli._produce(resolved.variant, resolved.backend, tmp_path, seed=0)
    assert out == tmp_path / "ckpt"
    assert seen == {
        "name": "int4-gptq",
        "backend": "vllm",
        "checkpoints_root": tmp_path,
        "calibration_seed": CALIBRATION_SEED,
        "seed": 0,
    }


def test_produce_rejects_a_backend_without_a_producer(tmp_path: Path) -> None:
    resolved = resolve_config(CONFIG_ROOT / "variants" / "fp16.yaml", config_root=CONFIG_ROOT)
    with pytest.raises(typer.BadParameter, match="no producer"):
        cli._produce(resolved.variant, resolved.backend, tmp_path, seed=0)


@pytest.mark.parametrize(("profile", "expected"), [(None, [0]), ("full-mmlu-seeds", [0, 1, 2])])
def test_run_produces_one_checkpoint_per_profile_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str | None, expected: list[int]
) -> None:
    seeds: list[int] = []

    def fake(
        _variant: VariantConfig,
        _backend: Mapping[str, Any],
        checkpoints: Path,
        *,
        seed: int,
        mode: RunMode,
    ) -> Path:
        assert mode == "full"
        seeds.append(seed)
        return checkpoints / f"ckpt-{seed}"

    monkeypatch.setattr(cli, "_produce", fake)
    args = ["run", "--config", str(CONFIG_ROOT / "variants" / "int4-gptq.yaml")]
    args += ["--checkpoints", str(tmp_path), "--config-root", str(CONFIG_ROOT)]
    if profile is not None:
        args += ["--eval", profile]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert seeds == expected


def test_torchao_ptq_has_nothing_to_produce(tmp_path: Path) -> None:
    resolved = resolve_config(
        CONFIG_ROOT / "variants" / "ptq-3bit-torchao.yaml", config_root=CONFIG_ROOT
    )
    with pytest.raises(typer.BadParameter, match="quantises on load"):
        cli._produce(resolved.variant, resolved.backend, tmp_path, seed=0)


@pytest.mark.parametrize(("mode", "cap"), [("smoke", cli.SMOKE_TRAIN_TOKENS), ("full", None)])
def test_qat_dispatches_to_the_trainer_with_the_smoke_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: RunMode, cap: int | None
) -> None:
    resolved = resolve_config(
        CONFIG_ROOT / "variants" / "qat-3bit-lora.yaml", mode=mode, config_root=CONFIG_ROOT
    )
    seen: dict[str, Any] = {}

    def fake(
        _variant: VariantConfig,
        _backend: Mapping[str, Any],
        *,
        checkpoints_root: Path,
        seed: int,
        device: str,
        max_tokens: int | None,
    ) -> Path:
        seen.update(seed=seed, device=device, max_tokens=max_tokens, root=checkpoints_root)
        return checkpoints_root / "qat"

    monkeypatch.setattr(cli, "produce_qat", fake)
    monkeypatch.setattr(cli, "resolve_device", lambda _mode: "cpu")
    out = cli._produce(resolved.variant, resolved.backend, tmp_path, seed=1, mode=mode)
    assert out == tmp_path / "qat"
    assert seen == {"seed": 1, "device": "cpu", "max_tokens": cap, "root": tmp_path}
