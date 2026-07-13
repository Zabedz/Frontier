"""Memory maths and the CPU capture path, no torch, no model, no network."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.latency.memory import (
    HFMemoryProbe,
    dtype_bytes,
    kv_cache_dims,
    kv_cache_mb,
    measure_peak_memory_mb,
    weights_disk_mb,
)
from frontier.latency.timing import TrialClock

CONTEXT_LEN = 128
DECODE_LEN = 8
FP16_KV_BYTES = 2
FP32_KV_BYTES = 4
INT8_KV_BYTES = 1


class _FakeDriver:
    def run_trial(
        self, clock: TrialClock, *, batch_size: int, context_len: int, decode_len: int
    ) -> None:
        del clock, batch_size, context_len, decode_len


class _FakeModel:
    def __init__(self, config: SimpleNamespace, dtype: str) -> None:
        self.config = config
        self.dtype = dtype


def test_kv_cache_mb_known_answer() -> None:
    result = kv_cache_mb(
        n_layers=32, n_kv_heads=8, head_dim=128, seq_len=1000, batch_size=1, dtype_bytes=2
    )
    # 2 * 32 * 8 * 128 * 1000 * 1 * 2 / 1e6.
    assert result == pytest.approx(131.072)


def test_kv_cache_dims_mha_and_head_dim_fallback() -> None:
    config = SimpleNamespace(num_hidden_layers=4, num_attention_heads=6, hidden_size=48)
    assert kv_cache_dims(config) == (4, 6, 8)


def test_kv_cache_dims_gqa_and_explicit_head_dim() -> None:
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=64,
        head_dim=16,
    )
    assert kv_cache_dims(config) == (4, 2, 16)


def test_dtype_bytes_table_and_unknown() -> None:
    assert dtype_bytes("fp16") == FP16_KV_BYTES
    assert dtype_bytes("int8") == INT8_KV_BYTES
    assert dtype_bytes("fp32") == FP32_KV_BYTES
    with pytest.raises(ValueError, match="unknown kv_cache_dtype"):
        dtype_bytes("int3")


def test_weights_disk_mb_sums_snapshot_files(tmp_path: Path) -> None:
    (tmp_path / "model-00001.safetensors").write_bytes(b"\0" * 1_000_000)
    (tmp_path / "model-00002.safetensors").write_bytes(b"\0" * 2_000_000)
    (tmp_path / "tokenizer.json").write_bytes(b"\0" * 500_000)
    assert weights_disk_mb(tmp_path) == pytest.approx(3.0)


def test_weights_disk_mb_none_is_zero() -> None:
    assert weights_disk_mb(None) == 0.0


def test_measure_peak_memory_cpu_is_finite() -> None:
    peak = measure_peak_memory_mb("cpu", None)
    assert math.isfinite(peak)
    assert peak >= 0.0


def test_hf_memory_probe_cpu_kv_matches_formula() -> None:
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=64,
        head_dim=8,
    )
    model = _FakeModel(config, dtype="torch.float32")
    probe = HFMemoryProbe(
        _FakeDriver(),
        model,
        device="cpu",
        decode_len=DECODE_LEN,
        model_dir=None,
        torch_module=None,
    )
    memory = probe.measure(batch_size=1, context_len=CONTEXT_LEN)

    assert memory.batch_size == 1
    assert memory.context_len == CONTEXT_LEN
    assert memory.weights_disk_mb == 0.0
    assert math.isfinite(memory.peak_vram_mb)
    assert math.isfinite(memory.weights_resident_mb)
    expected_kv = kv_cache_mb(
        n_layers=4,
        n_kv_heads=2,
        head_dim=8,
        seq_len=CONTEXT_LEN + DECODE_LEN,
        batch_size=1,
        dtype_bytes=FP32_KV_BYTES,
    )
    assert memory.kv_cache_mb == pytest.approx(expected_kv)
