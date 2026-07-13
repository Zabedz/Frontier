"""Peak-memory capture, weights on disk vs resident, and the analytic KV-cache size.

Three memory numbers, measured differently. Peak VRAM is read from the CUDA allocator
on the pod and proxied by process RSS on a laptop (labelled a proxy below, kept finite
so ``tok_s_per_gb`` is finite on a smoke row; the ``cpu:`` hardware_id makes the row
self-describing). Weights-on-disk is summed from the snapshot files. The KV-cache size
is analytic from the model config, so it is CPU-testable, and it is the clean
"KV growth with context length" curve rather than a noisy measurement.

torch is lazy and reached only as an injected module; ``resource``/``sys`` carry the
CPU path. On CPU the compute dtype is float32, so the KV-cache size uses the model's own
parameter dtype rather than the configured GPU ``kv_cache_dtype``.
"""

from __future__ import annotations

import resource
import sys
from pathlib import Path
from typing import Any, Protocol

from frontier.latency.timing import GenerationDriver
from frontier.schema import Memory

_KEYS_AND_VALUES = 2
_BYTES_PER_MB = 1e6
_LINUX_RUSAGE_KIB = 1024.0
_DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "float16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
    "int8": 1,
}
_WEIGHT_GLOBS = ("*.safetensors", "*.bin")


def kv_cache_mb(
    *,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int,
) -> float:
    """Analytic KV-cache footprint in decimal MB.

    ``2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes / 1e6``,
    the factor 2 being keys and values. ``seq_len`` is defined as ``context_len +
    decode_len`` (the footprint of the context plus the generated tokens); the literal
    per-step peak is one position lower, since the final generated token is not fed back.
    """
    elements = _KEYS_AND_VALUES * n_layers * n_kv_heads * head_dim * seq_len * batch_size
    return elements * dtype_bytes / _BYTES_PER_MB


def kv_cache_dims(config: Any) -> tuple[int, int, int]:
    """Pull ``(n_layers, n_kv_heads, head_dim)`` from a HF config.

    ``num_hidden_layers``; ``num_key_value_heads`` when present else
    ``num_attention_heads`` (the MHA fallback); ``head_dim`` when present else
    ``hidden_size // num_attention_heads``.
    """
    n_layers = int(config.num_hidden_layers)
    n_heads = int(config.num_attention_heads)
    n_kv_heads = int(getattr(config, "num_key_value_heads", None) or n_heads)
    head_dim = int(getattr(config, "head_dim", None) or (int(config.hidden_size) // n_heads))
    return n_layers, n_kv_heads, head_dim


def dtype_bytes(kv_cache_dtype: str) -> int:
    """Bytes per KV element: fp16/bf16 to 2, fp32 to 4, int8 to 1. Raises on unknown."""
    try:
        return _DTYPE_BYTES[kv_cache_dtype]
    except KeyError:
        raise ValueError(
            f"unknown kv_cache_dtype {kv_cache_dtype!r}; expected one of {sorted(_DTYPE_BYTES)}"
        ) from None


def weights_disk_mb(model_dir: Path | None) -> float:
    """Sum of ``*.safetensors`` and ``*.bin`` sizes under the snapshot dir, decimal MB.

    ``model_dir`` is the resolved HF cache snapshot path, or ``None`` when unknown (then
    ``0.0``). Track-B single-file GGUF weights are a later WP.
    """
    if model_dir is None:
        return 0.0
    total = sum(
        path.stat().st_size for pattern in _WEIGHT_GLOBS for path in model_dir.glob(pattern)
    )
    return total / _BYTES_PER_MB


def measure_peak_memory_mb(device: str, torch_module: Any | None) -> float:
    """Peak allocation in decimal MB.

    On ``cuda``: ``torch.cuda.max_memory_allocated / 1e6`` (the caller resets the peak
    before the measured generation). On ``cpu``: a process-RSS proxy via
    ``resource.getrusage`` max RSS, unit-corrected (bytes on Darwin, KiB on Linux). The
    CPU value is process RSS, not device VRAM; it is kept so the field stays finite.
    """
    if device.startswith("cuda") and torch_module is not None:
        return float(torch_module.cuda.max_memory_allocated(device)) / _BYTES_PER_MB
    return _process_rss_mb()


class MemoryProbe(Protocol):
    """Measures one ``schema.Memory`` per (batch size, context length)."""

    def measure(self, *, batch_size: int, context_len: int) -> Memory: ...


class HFMemoryProbe:
    """Measures one ``schema.Memory`` per (batch, context) on a loaded model.

    Resets the CUDA peak, runs one generation at (batch, context) through the driver,
    reads ``measure_peak_memory_mb`` for ``peak_vram_mb``, and computes ``kv_cache_mb``
    analytically at ``seq_len = context_len + decode_len``. ``weights_disk_mb`` comes from
    the snapshot dir and ``weights_resident_mb`` from ``torch.cuda.memory_allocated``
    captured at construction, right after load (an RSS proxy on CPU). The KV-cache dtype
    is the model's own parameter dtype, which is float32 on the CPU path.
    """

    def __init__(
        self,
        driver: GenerationDriver,
        model: Any,
        *,
        device: str,
        decode_len: int,
        model_dir: Path | None,
        torch_module: Any | None,
    ) -> None:
        self._driver = driver
        self._device = device
        self._decode_len = decode_len
        self._torch = torch_module
        self._weights_disk_mb = weights_disk_mb(model_dir)
        self._weights_resident_mb = _resident_memory_mb(device, torch_module)
        self._n_layers, self._n_kv_heads, self._head_dim = kv_cache_dims(model.config)
        self._kv_dtype_bytes = dtype_bytes(_model_dtype_name(model))

    def measure(self, *, batch_size: int, context_len: int) -> Memory:
        if self._device.startswith("cuda") and self._torch is not None:
            self._torch.cuda.reset_peak_memory_stats(self._device)  # pragma: no cover
        self._driver.run_trial(
            _DiscardClock(),
            batch_size=batch_size,
            context_len=context_len,
            decode_len=self._decode_len,
        )
        kv = kv_cache_mb(
            n_layers=self._n_layers,
            n_kv_heads=self._n_kv_heads,
            head_dim=self._head_dim,
            seq_len=context_len + self._decode_len,
            batch_size=batch_size,
            dtype_bytes=self._kv_dtype_bytes,
        )
        return Memory(
            batch_size=batch_size,
            context_len=context_len,
            peak_vram_mb=measure_peak_memory_mb(self._device, self._torch),
            weights_disk_mb=self._weights_disk_mb,
            weights_resident_mb=self._weights_resident_mb,
            kv_cache_mb=kv,
        )


class _DiscardClock:
    """The clock the memory probe hands the driver; its timings are not read here."""

    def mark(self) -> None:
        return None

    def resolve(self) -> list[float]:
        return []


def _resident_memory_mb(device: str, torch_module: Any | None) -> float:
    if device.startswith("cuda") and torch_module is not None:
        return float(torch_module.cuda.memory_allocated(device)) / _BYTES_PER_MB
    return _process_rss_mb()


def _process_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    factor = 1.0 if sys.platform == "darwin" else _LINUX_RUSAGE_KIB
    return rss * factor / _BYTES_PER_MB


def _model_dtype_name(model: Any) -> str:
    return str(getattr(model, "dtype", "torch.float16")).rsplit(".", maxsplit=1)[-1]
