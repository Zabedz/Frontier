"""Hand-built ``ResultRow`` fixtures for the io round-trip tests.

The finite builder feeds the exact-equality round-trips; the NaN and populated-latency
builders cover the deferred fields and the JSON-list encoding before the latency rig
exists. Kept as plain functions (not fixtures) so a test can build several rows.
"""

from __future__ import annotations

import math

from frontier.schema import (
    Backend,
    Latency,
    MachineState,
    Memory,
    Provenance,
    Quality,
    ResultRow,
    Robustness,
    TaskSpec,
)


def sample_provenance(seed: int = 0) -> Provenance:
    return Provenance(
        git_sha="deadbeef",
        config_hash="0" * 64,
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        model_revision="main",
        hardware_id="cpu:arm64",
        driver_version="none",
        cuda_version="none",
        seed=seed,
        timestamp="2026-07-13T00:00:00+00:00",
    )


def sample_backend() -> Backend:
    return Backend(
        inference_backend="hf",
        backend_version="5.0.0",
        weight_dtype="fp16",
        kv_cache_dtype="fp16",
        gpu_offload_layers=-1,
        track="A",
    )


def sample_task() -> TaskSpec:
    return TaskSpec(
        task_name="mmlu",
        split="test",
        num_items=50,
        prompt_style="zeroshot",
        scoring="letter_softmax",
        permutation_scheme="cyclic",
        labels="raw",
        cot=False,
    )


def sample_quality(*, perplexity: float = 12.3) -> Quality:
    return Quality(
        accuracy=0.82,
        accuracy_ci_low=0.78,
        accuracy_ci_high=0.86,
        ece_equal_width=0.041,
        ece_equal_mass_ace=0.037,
        ece_bin_sweep={5: 0.03, 10: 0.041, 15: 0.052},
        ece_ci_low=0.02,
        ece_ci_high=0.06,
        brier=0.24,
        brier_reliability=0.018,
        brier_resolution=0.15,
        brier_uncertainty=0.37,
        nll=0.51,
        perplexity=perplexity,
        temperature_scaled=False,
        temperature=1.0,
    )


def sample_row(
    *,
    seed: int = 0,
    with_robustness: bool = True,
    perplexity: float = 12.3,
    tok_s_per_gb: float = 5.0,
) -> ResultRow:
    """A fully populated, finite row (empty latency/memory), for exact-equality tests."""
    robustness = Robustness(0.91, 0.06, 0.12) if with_robustness else None
    return ResultRow(
        provenance=sample_provenance(seed),
        backend=sample_backend(),
        variant_name="fp16",
        family="baseline",
        task=sample_task(),
        quality=sample_quality(perplexity=perplexity),
        latency=[],
        memory=[],
        tok_s_per_gb=tok_s_per_gb,
        robustness=robustness,
    )


def nan_row() -> ResultRow:
    """A row with the WP3 deferred NaN fields, checked with ``math.isnan`` after reload."""
    return sample_row(perplexity=math.nan, tok_s_per_gb=math.nan)


def row_with_latency() -> ResultRow:
    """A row with a populated ``latency``/``memory`` list, to prove the JSON encoding.

    The latency rig does not exist yet, so this is hand-built to show the nested
    ``Latency`` + ``MachineState`` and ``Memory`` round-trip through the JSON columns.
    """
    machine = MachineState(
        gpu_clock_sm_mhz=1500,
        gpu_clock_mem_mhz=6000,
        gpu_temp_c=61,
        power_w=118.5,
        clocks_locked=False,
        clock_drift_flag=False,
    )
    latency = Latency(
        batch_size=1,
        ttft_median_ms=10.2,
        ttft_p95_ms=12.7,
        itl_median_ms=3.1,
        itl_p95_ms=4.4,
        throughput_tok_s=203.0,
        n_trials=20,
        warmup_discarded=5,
        machine_state=machine,
    )
    memory = Memory(
        batch_size=1,
        context_len=512,
        peak_vram_mb=8100.0,
        weights_disk_mb=3000.0,
        weights_resident_mb=3200.0,
        kv_cache_mb=410.0,
    )
    base = sample_row()
    return ResultRow(
        provenance=base.provenance,
        backend=base.backend,
        variant_name=base.variant_name,
        family=base.family,
        task=base.task,
        quality=base.quality,
        latency=[latency],
        memory=[memory],
        tok_s_per_gb=base.tok_s_per_gb,
        robustness=base.robustness,
    )
