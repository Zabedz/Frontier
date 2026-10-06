"""The CLI wiring: mode parsing and the typer command surface, model-free."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from frontier.pipeline.cli import app, parse_mode

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
FP16 = CONFIG_ROOT / "variants" / "fp16.yaml"


def testparse_mode_accepts_the_two_modes() -> None:
    assert parse_mode("smoke") == "smoke"
    assert parse_mode("full") == "full"


def testparse_mode_rejects_other_values() -> None:
    with pytest.raises(typer.BadParameter):
        parse_mode("turbo")


def test_cli_rejects_bad_mode_before_loading() -> None:
    result = CliRunner().invoke(app, ["run", "--config", str(FP16), "--mode", "turbo"])
    assert result.exit_code != 0


def test_cli_rejects_missing_config() -> None:
    result = CliRunner().invoke(app, ["run", "--config", str(CONFIG_ROOT / "nope.yaml")])
    assert result.exit_code != 0
