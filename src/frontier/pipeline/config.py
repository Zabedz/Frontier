"""Config resolution: deep-merge ``base.yaml``, the variant file, the eval profile, and
``smoke.yaml`` in that order, validate against ``variant.schema.json``, type, and hash.

The hash is taken after all overlays, so a smoke run and a full run of one variant get
different hashes and the hash identifies the exact config that produced a row.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from frontier.schema import (
    DistillSpec,
    EvalSpec,
    LatencySpec,
    ModelSpec,
    QuantSpec,
    RunMode,
    VariantConfig,
)

DEFAULT_CONFIG_ROOT = Path("configs")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, mutating neither.

    Nested dicts merge key by key; any non-dict value from ``override``, a list included,
    replaces the base value wholesale, so smoke's ``seeds: [0]`` replaces the base list.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def config_hash(resolved: Mapping[str, Any]) -> str:
    """SHA-256 hex of the canonical JSON of the resolved config.

    ``sort_keys`` makes the hash invariant to the key order the merge produced; list order
    is preserved because it is meaningful (``seeds``).
    """
    canonical = json.dumps(
        resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """A fully merged, validated run configuration.

    ``raw`` is the merged dict the hash is taken over and the schema validated against;
    ``eval_spec`` is ``variant.eval``; ``backend`` is the raw ``backend`` block, which the
    frozen ``VariantConfig`` has no field for.
    """

    raw: Mapping[str, Any]
    variant: VariantConfig
    eval_spec: EvalSpec
    backend: Mapping[str, Any]
    mode: RunMode
    config_hash: str


def resolve_config(
    config_path: Path,
    *,
    eval_profile: str | None = None,
    mode: RunMode = "full",
    config_root: Path = DEFAULT_CONFIG_ROOT,
) -> ResolvedConfig:
    """Merge base + variant + eval-profile + smoke, validate, type, and hash."""
    merged = deep_merge(_load_yaml(config_root / "base.yaml"), _load_yaml(config_path))
    if eval_profile is not None:
        merged = deep_merge(merged, _load_yaml(config_root / "evals" / f"{eval_profile}.yaml"))
    if mode == "smoke":
        merged = deep_merge(merged, _load_yaml(config_root / "smoke.yaml"))

    jsonschema.validate(merged, _load_schema(config_root))
    variant = _to_variant_config(merged)
    return ResolvedConfig(
        raw=merged,
        variant=variant,
        eval_spec=variant.eval,
        backend=merged.get("backend", {}),
        mode=mode,
        config_hash=config_hash(merged),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"config layer {path} did not parse to a mapping, got {type(loaded)}")
    return loaded


def _load_schema(config_root: Path) -> dict[str, Any]:
    with (config_root / "schema" / "variant.schema.json").open(encoding="utf-8") as handle:
        schema: dict[str, Any] = json.load(handle)
    return schema


def _to_variant_config(raw: Mapping[str, Any]) -> VariantConfig:
    quant = raw.get("quant")
    distill = raw.get("distill")
    return VariantConfig(
        name=raw["name"],
        family=raw["family"],
        track=raw["track"],
        model=ModelSpec(**raw.get("model", {})),
        eval=_to_eval_spec(raw.get("eval", {})),
        latency=_to_latency_spec(raw.get("latency", {})),
        quant=QuantSpec(**quant) if quant else None,
        distill=DistillSpec(**distill) if distill else None,
    )


def _to_eval_spec(block: Mapping[str, Any]) -> EvalSpec:
    data = dict(block)
    if "seeds" in data:
        data["seeds"] = tuple(data["seeds"])
    return EvalSpec(**data)


def _to_latency_spec(block: Mapping[str, Any]) -> LatencySpec:
    data = dict(block)
    for key in ("batch_sizes", "context_lengths"):
        if key in data:
            data[key] = tuple(data[key])
    return LatencySpec(**data)
