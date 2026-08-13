"""Contract tests for the scaffold: the schema dataclasses and the variant configs on disk."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jsonschema
import pytest
import yaml

import frontier
from frontier.schema import Backend, Provenance, ResultRow

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
VARIANT_CONFIGS = sorted((CONFIG_DIR / "variants").glob("*.yaml"))


def test_package_has_docstring() -> None:
    assert frontier.__doc__ is not None


def test_resultrow_leads_with_provenance_then_backend() -> None:
    fields = dataclasses.fields(ResultRow)
    assert fields[0].name == "provenance"
    assert fields[1].name == "backend"
    for required in (fields[0], fields[1]):
        assert required.default is dataclasses.MISSING
        assert required.default_factory is dataclasses.MISSING


def test_resultrow_cannot_be_constructed_empty() -> None:
    with pytest.raises(TypeError):
        ResultRow()  # type: ignore[call-arg]


@pytest.mark.parametrize("record", [Provenance, Backend])
def test_provenance_and_backend_have_no_optional_fields(record: type) -> None:
    for spec in dataclasses.fields(record):
        assert spec.default is dataclasses.MISSING
        assert spec.default_factory is dataclasses.MISSING


def _load_schema() -> dict[str, object]:
    with (CONFIG_DIR / "schema" / "variant.schema.json").open() as handle:
        schema: dict[str, object] = json.load(handle)
    return schema


def test_variant_configs_exist() -> None:
    assert VARIANT_CONFIGS, "no variant configs found under configs/variants/"


@pytest.mark.parametrize("config_path", VARIANT_CONFIGS, ids=lambda p: p.name)
def test_variant_config_validates_against_schema(config_path: Path) -> None:
    schema = _load_schema()
    with config_path.open() as handle:
        data = yaml.safe_load(handle)
    jsonschema.validate(data, schema)
    assert {"name", "family", "track"} <= set(data)
