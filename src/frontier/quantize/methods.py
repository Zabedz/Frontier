"""The torchao quant-method names, in one import-free place for the registry, the checkpoint
paths, and the QAT producer."""

from __future__ import annotations

TORCHAO_PTQ_METHOD = "torchao-intx"
QAT_LORA_METHOD = "torchao-qat-lora"
QAT_FULL_METHOD = "torchao-qat-full"
QAT_METHODS = frozenset({QAT_LORA_METHOD, QAT_FULL_METHOD})


def is_qat_method(method: str) -> bool:
    """True when ``method`` names a torchao QAT recipe that writes a checkpoint."""
    return method in QAT_METHODS
