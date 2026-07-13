"""Inference backends: one per way of loading a model, all satisfying the WP2
``LogitProvider`` seam. The two-track design means several backends (HF now; vLLM,
llama.cpp, torchao later) that read logits the same way but load models differently,
so each lives here rather than tangled into the eval core or the runner. The HF
Track-A backend imports ``transformers`` lazily, so importing this package is cheap
and does not need the ``hf`` dependency group.
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

__all__ = [
    "DEFAULT_REVISION",
    "HFLogitProvider",
    "chat_wrap",
    "resolve_candidates",
    "resolve_device",
    "resolve_dtype",
]
