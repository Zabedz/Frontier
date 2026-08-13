"""Inference backends, one per way of loading a model, all satisfying the ``LogitProvider``
seam. Each imports its heavy stack lazily, so importing this package needs none of them.
"""

from __future__ import annotations

from frontier.backends.hf import (
    DEFAULT_REVISION,
    HFLogitProvider,
    chat_wrap,
    resolve_candidates,
    resolve_device,
    resolve_dtype,
)
from frontier.backends.llama_cpp import LlamaCppLogitProvider
from frontier.backends.registry import build_latency_probe, build_provider
from frontier.backends.vllm import VllmLogitProvider

__all__ = [
    "DEFAULT_REVISION",
    "HFLogitProvider",
    "LlamaCppLogitProvider",
    "VllmLogitProvider",
    "build_latency_probe",
    "build_provider",
    "chat_wrap",
    "resolve_candidates",
    "resolve_device",
    "resolve_dtype",
]
