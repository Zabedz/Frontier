"""The config-to-row runner and the ``frontier`` CLI: resolve a variant config in smoke or
full mode, run the eval and latency stages, stamp provenance, append one result row.
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
