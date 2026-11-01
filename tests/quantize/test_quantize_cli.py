"""The ``frontier-quantize`` dispatch: producer selection by family and backend, model-free."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from frontier.distill.build import CUBLAS_WORKSPACE_ENV, DETERMINISTIC_CUBLAS_WORKSPACE
from frontier.distill.cache import write_json
from frontier.distill.fidelity import FIDELITY_FILE, FidelityReport
from frontier.distill.heldout import BASE_HELDOUT_FILE, HeldoutReport, write_heldout
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
        max_tokens: int | None,
    ) -> Path:
        assert mode == "full"
        assert max_tokens is None
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


@pytest.mark.parametrize(
    ("mode", "flag", "expected"),
    [("full", 491_520, 491_520), ("smoke", 2048, 2048), ("smoke", None, cli.SMOKE_TRAIN_TOKENS)],
)
def test_max_tokens_overrides_the_mode_default(
    mode: RunMode, flag: int | None, expected: int | None
) -> None:
    assert cli._token_cap(mode, flag) == expected


@pytest.mark.parametrize(("mode", "cap"), [("smoke", cli.SMOKE_TRAIN_TOKENS), ("full", None)])
def test_distill_dispatches_to_the_trainer_with_the_smoke_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: RunMode, cap: int | None
) -> None:
    resolved = resolve_config(
        CONFIG_ROOT / "variants" / "distill-corpus.yaml", mode=mode, config_root=CONFIG_ROOT
    )
    seen: dict[str, Any] = {}

    def fake(
        variant: VariantConfig,
        *,
        checkpoints_root: Path,
        seed: int,
        device: str,
        max_tokens: int | None,
    ) -> Path:
        seen.update(name=variant.name, seed=seed, device=device, max_tokens=max_tokens)
        return checkpoints_root / "student"

    monkeypatch.setattr(cli, "produce_distill", fake)
    monkeypatch.setattr(cli, "resolve_device", lambda _mode: "cpu")
    _unset_cublas(monkeypatch)
    out = cli._produce(resolved.variant, resolved.backend, tmp_path, seed=2, mode=mode)
    assert out == tmp_path / "student"
    assert seen == {"name": "distill-corpus", "seed": 2, "device": "cpu", "max_tokens": cap}
    assert os.environ[CUBLAS_WORKSPACE_ENV] == DETERMINISTIC_CUBLAS_WORKSPACE


def _unset_cublas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset for the test and restored unset after, so later tests never inherit it."""
    monkeypatch.setenv(CUBLAS_WORKSPACE_ENV, "")
    monkeypatch.delenv(CUBLAS_WORKSPACE_ENV)


def test_cache_refuses_a_variant_without_a_distill_block(tmp_path: Path) -> None:
    args = ["cache", "--config", str(CONFIG_ROOT / "variants" / "fp16.yaml")]
    args += ["--checkpoints", str(tmp_path), "--config-root", str(CONFIG_ROOT)]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code != 0
    assert "no distill block" in result.output


def test_cache_builds_at_the_capped_budget_and_prints_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    meta = SimpleNamespace(
        n_rows=20,
        seq_len=1024,
        top_k=64,
        teacher_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        teacher_dtype="float32",
        device="cpu",
        deterministic=True,
        rest_summary={"mean": 0.1, "p50": 0.05, "p99": 0.6, "max": 0.9},
    )
    report = FidelityReport(
        rows=(0, 1, 2, 3),
        positions=4 * 1023,
        prob_error_ratio=0.0,
        prob_abs_error=0.0,
        rest_error_ratio=0.0,
        rest_abs_error=0.0,
        topk_violations=0,
        order_violations=0,
        duplicate_ids=0,
        next_token_agreement={0: 0.01, 1: 0.6, 2: 0.05},
        rest_mean=0.1,
        rest_max=0.9,
        first_failure=None,
    )

    def fake(
        variant: VariantConfig,
        *,
        checkpoints_root: Path,
        device: str,
        max_tokens: int | None,
    ) -> Any:
        seen.update(name=variant.name, root=checkpoints_root, device=device, max_tokens=max_tokens)
        write_json(tmp_path / FIDELITY_FILE, {**asdict(report), "passed": True})
        write_heldout(
            tmp_path / BASE_HELDOUT_FILE,
            HeldoutReport("HuggingFaceTB/SmolLM2-135M-Instruct", 4, 4092, 0.5, 2.0, 3.0, 0.4),
        )
        return SimpleNamespace(path=tmp_path, meta=meta, training=range(16), heldout=range(16, 20))

    monkeypatch.setattr(cli, "ensure_cache", fake)
    monkeypatch.setattr(cli, "resolve_device", lambda _mode: "cpu")
    _unset_cublas(monkeypatch)
    args = ["cache", "--config", str(CONFIG_ROOT / "variants" / "distill-softlabel-topk.yaml")]
    args += ["--checkpoints", str(tmp_path), "--mode", "smoke", "--config-root", str(CONFIG_ROOT)]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert seen == {
        "name": "distill-softlabel-topk",
        "root": tmp_path,
        "device": "cpu",
        "max_tokens": cli.SMOKE_TRAIN_TOKENS,
    }
    printed = result.output.replace("\n", "")
    assert str(tmp_path) in printed
    assert "1: 0.600" in printed
