"""The config-to-row runner and the CLI. Resolves a variant config (smoke or full
mode), dispatches to the right quantise / eval / latency stages, stamps provenance,
and appends one result row. Adding a variant is adding a config file; this package
does not change per variant.
"""

from __future__ import annotations

from frontier.pipeline.config import (
    ResolvedConfig,
    config_hash,
    deep_merge,
    resolve_config,
)
from frontier.pipeline.runner import load_slice, run

__all__ = [
    "ResolvedConfig",
    "config_hash",
    "deep_merge",
    "load_slice",
    "resolve_config",
    "run",
]
