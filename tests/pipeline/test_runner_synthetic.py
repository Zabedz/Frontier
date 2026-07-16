"""The full config -> score -> metrics -> assemble -> store chain, offline.

A synthetic provider and in-memory records drive the runner with no model and no
network, proving the wiring the live smoke test exercises with a real model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from frontier.eval.provider import LogitProvider
from frontier.eval.records import LETTERS, EvalRecord
from frontier.io.predictions import predictions_key, predictions_path, read_predictions
from frontier.io.store import ResultStore, read_rows
from frontier.latency.rig import LatencyMemory
from frontier.pipeline.config import ResolvedConfig
from frontier.pipeline.runner import run
from frontier.schema import EvalSpec, Latency, MachineState, Memory, RunMode, VariantConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
FP16 = CONFIG_ROOT / "variants" / "fp16.yaml"
GOLD_MARK = "<<GOLD>>"
CANNED_SM_MHZ = 1500
N_ITEMS = 16
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
_LETTER_INDEX = {letter: i for i, letter in enumerate(LETTERS)}


class _SyntheticProvider:
    """Favours the gold option, which each record marks in its option text.

    Carries ``backend_version`` so ``_build_backend`` reads a real value off the
    provider, exactly as the HF backend supplies it.
    """

    backend_version = "synthetic-1.0"

    def candidate_token_ids(self, letters: Sequence[str]) -> IntArray:
        return np.arange(len(letters), dtype=np.intp)

    def next_token_logits(self, prompts: Sequence[str]) -> FloatArray:
        return np.stack([self._logits(prompt) for prompt in prompts])

    def _logits(self, prompt: str) -> FloatArray:
        row = np.full(len(LETTERS), -10.0, dtype=np.float64)
        position = 0
        for line in prompt.splitlines():
            if line[1:2] == "." and line[:1] in _LETTER_INDEX:
                row[position] = 2.0 if GOLD_MARK in line else 0.0
                position += 1
        return row


def _canned_latency(
    provider: LogitProvider,  # noqa: ARG001
    resolved: ResolvedConfig,  # noqa: ARG001
    *,
    device: str,  # noqa: ARG001
    mode: RunMode,  # noqa: ARG001
) -> LatencyMemory:
    """A model-free, network-free stand-in for the real latency rig."""
    machine = MachineState(
        gpu_clock_sm_mhz=CANNED_SM_MHZ,
        gpu_clock_mem_mhz=6000,
        gpu_temp_c=61,
        power_w=118.5,
        clocks_locked=False,
        clock_drift_flag=False,
    )
    latency = Latency(
        batch_size=1,
        ttft_median_ms=10.0,
        ttft_p95_ms=12.0,
        itl_median_ms=3.0,
        itl_p95_ms=4.0,
        throughput_tok_s=200.0,
        n_trials=3,
        warmup_discarded=1,
        machine_state=machine,
    )
    memory = Memory(
        batch_size=1,
        context_len=128,
        peak_vram_mb=900.0,
        weights_disk_mb=500.0,
        weights_resident_mb=520.0,
        kv_cache_mb=12.0,
    )
    return LatencyMemory(latency=[latency], memory=[memory], tok_s_per_gb=222.0)


def _records(n: int) -> list[EvalRecord]:
    records = []
    for i in range(n):
        gold = i % 4
        options = tuple(f"choice {j} {GOLD_MARK}" if j == gold else f"choice {j}" for j in range(4))
        records.append(
            EvalRecord(
                qid=f"q{i}", question="Q?", options=options, gold=gold, subject="sub", split="test"
            )
        )
    return records


def test_synthetic_runner_emits_one_valid_row(tmp_path: Path) -> None:
    def factory(variant: VariantConfig, device: str) -> _SyntheticProvider:  # noqa: ARG001
        return _SyntheticProvider()

    def loader(spec: EvalSpec, *, seed: int) -> list[EvalRecord]:  # noqa: ARG001
        return _records(16)

    rows = run(
        FP16,
        mode="smoke",
        config_root=CONFIG_ROOT,
        results_root=tmp_path,
        provider_factory=factory,
        slice_loader=loader,
        timestamp="2026-07-13T00:00:00+00:00",
        git_sha="deadbeef",
        latency_probe=_canned_latency,
    )

    assert len(rows) == 1
    row = rows[0]

    assert row.provenance.git_sha == "deadbeef"
    assert row.provenance.timestamp == "2026-07-13T00:00:00+00:00"
    assert row.provenance.model_id == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert row.provenance.config_hash
    assert row.provenance.hardware_id.startswith("cpu:")

    assert row.backend.inference_backend == "hf"
    assert row.backend.backend_version == "synthetic-1.0"
    assert row.backend.track == "A"

    assert 0.0 <= row.quality.accuracy <= 1.0
    assert np.isfinite(row.quality.ece_equal_width)
    assert 0.0 <= row.quality.ece_equal_width <= 1.0
    assert math.isnan(row.quality.perplexity)

    assert row.robustness is not None
    assert row.latency
    assert row.latency[0].machine_state is not None
    assert row.memory
    assert math.isfinite(row.tok_s_per_gb)

    stored = read_rows(ResultStore(tmp_path))
    assert len(stored) == 1
    assert stored[0].backend.backend_version == "synthetic-1.0"
    assert stored[0].provenance.git_sha == "deadbeef"
    assert math.isnan(stored[0].quality.perplexity)
    assert stored[0].latency[0].machine_state.gpu_clock_sm_mhz == CANNED_SM_MHZ

    key = predictions_key(row.provenance.config_hash, row.provenance.seed, row.task.task_name)
    sidecar = predictions_path(tmp_path, key)
    assert sidecar.exists()
    preds = read_predictions(tmp_path, key)
    assert preds.confidence.shape[0] == N_ITEMS
    assert bool(((preds.confidence >= 0.0) & (preds.confidence <= 1.0)).all())
    # ``correct`` is exactly ``exact_match(predicted, gold)``, so it reconstructs off
    # the self-describing sidecar columns.
    assert np.array_equal(preds.correct, preds.predicted == preds.gold)


def test_synthetic_runner_resumes_skipping_done_seeds(tmp_path: Path) -> None:
    def factory(variant: VariantConfig, device: str) -> _SyntheticProvider:  # noqa: ARG001
        return _SyntheticProvider()

    def loader(spec: EvalSpec, *, seed: int) -> list[EvalRecord]:  # noqa: ARG001
        return _records(N_ITEMS)

    first = run(
        FP16,
        mode="smoke",
        config_root=CONFIG_ROOT,
        results_root=tmp_path,
        provider_factory=factory,
        slice_loader=loader,
        timestamp="2026-07-13T00:00:00+00:00",
        git_sha="deadbeef",
        latency_probe=_canned_latency,
    )
    assert len(first) == 1

    def exploding_factory(variant: VariantConfig, device: str) -> _SyntheticProvider:  # noqa: ARG001
        raise AssertionError("the provider must not be built when the run is already complete")

    second = run(
        FP16,
        mode="smoke",
        config_root=CONFIG_ROOT,
        results_root=tmp_path,
        provider_factory=exploding_factory,
        slice_loader=loader,
        timestamp="2026-07-13T00:00:00+00:00",
        git_sha="deadbeef",
        latency_probe=_canned_latency,
    )
    assert second == []
    assert len(read_rows(ResultStore(tmp_path))) == 1


def test_synthetic_runner_skips_latency_when_disabled(tmp_path: Path) -> None:
    def factory(variant: VariantConfig, device: str) -> _SyntheticProvider:  # noqa: ARG001
        return _SyntheticProvider()

    def loader(spec: EvalSpec, *, seed: int) -> list[EvalRecord]:  # noqa: ARG001
        return _records(16)

    rows = run(
        FP16,
        mode="smoke",
        config_root=CONFIG_ROOT,
        results_root=tmp_path,
        provider_factory=factory,
        slice_loader=loader,
        timestamp="2026-07-13T00:00:00+00:00",
        git_sha="deadbeef",
        measure_latency=False,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.latency == []
    assert row.memory == []
    assert math.isnan(row.tok_s_per_gb)


def test_synthetic_runner_skips_predictions_when_disabled(tmp_path: Path) -> None:
    def factory(variant: VariantConfig, device: str) -> _SyntheticProvider:  # noqa: ARG001
        return _SyntheticProvider()

    def loader(spec: EvalSpec, *, seed: int) -> list[EvalRecord]:  # noqa: ARG001
        return _records(N_ITEMS)

    run(
        FP16,
        mode="smoke",
        config_root=CONFIG_ROOT,
        results_root=tmp_path,
        provider_factory=factory,
        slice_loader=loader,
        timestamp="2026-07-13T00:00:00+00:00",
        git_sha="deadbeef",
        latency_probe=_canned_latency,
        write_predictions=False,
    )

    assert not (tmp_path / "predictions").exists()
